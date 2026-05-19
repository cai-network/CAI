// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
use crate::ext::MultiaddrExt;
use delegate::delegate;
use either::Either;
use futures_lite::FutureExt;
use futures_timer::Delay;
use libp2p::core::transport::PortUse;
use libp2p::core::{ConnectedPoint, Endpoint};
use libp2p::swarm::behaviour::ConnectionEstablished;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{
    CloseConnection, ConnectionClosed, ConnectionDenied, ConnectionHandler,
    ConnectionHandlerSelect, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent,
    THandlerOutEvent, ToSwarm, dummy,
};
use libp2p::{Multiaddr, PeerId, identity, mdns};
use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::io;
use std::net::IpAddr;
use std::task::{Context, Poll};
use std::time::Duration;
use util::wakerdeque::WakerDeque;

const RETRY_CONNECT_INTERVAL: Duration = Duration::from_secs(5);

mod managed {
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{identity, mdns, ping};
    use std::io;
    use std::time::Duration;

    const MDNS_RECORD_TTL: Duration = Duration::from_secs(2_500);
    const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);
    const PING_TIMEOUT: Duration = Duration::from_millis(2_500);
    const PING_INTERVAL: Duration = Duration::from_millis(2_500);

    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        mdns: mdns::tokio::Behaviour,
        ping: ping::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
            Ok(Self {
                mdns: mdns_behaviour(keypair)?,
                ping: ping_behaviour(),
            })
        }
    }

    fn mdns_behaviour(keypair: &identity::Keypair) -> io::Result<mdns::tokio::Behaviour> {
        use mdns::{Config, tokio};

        // mDNS config => enable IPv6
        let mdns_config = Config {
            ttl: MDNS_RECORD_TTL,
            query_interval: MDNS_QUERY_INTERVAL,

            // enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
            ..Default::default()
        };

        let mdns_behaviour = tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id());
        Ok(mdns_behaviour?)
    }

    fn ping_behaviour() -> ping::Behaviour {
        ping::Behaviour::new(
            ping::Config::new()
                .with_timeout(PING_TIMEOUT)
                .with_interval(PING_INTERVAL),
        )
    }
}

/// Events for when a listening connection is truly established and truly closed.
#[derive(Debug, Clone)]
pub enum Event {
    ConnectionEstablished {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
    ConnectionClosed {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
}

/// Discovery behavior that wraps mDNS to produce truly discovered durable peer-connections.
///
/// The behaviour operates as such:
///  1) All true (listening) connections/disconnections are tracked, emitting corresponding events
///     to the swarm.
///  1) mDNS discovered/expired peers are tracked; discovered but not connected peers are dialed
///     immediately, and expired but connected peers are disconnected from immediately.
///  2) Every fixed interval: discovered but not connected peers are dialed, and expired but
///     connected peers are disconnected from.
pub struct Behaviour {
    // state-tracking for managed behaviors & mDNS-discovered peers
    managed: managed::Behaviour,
    mdns_discovered: HashMap<PeerId, BTreeSet<Multiaddr>>,
    bootstrap_peers: Vec<Multiaddr>,
    connected_peers: HashMap<PeerId, usize>,

    retry_delay: Delay, // retry interval

    // pending events to emmit => waker-backed Deque to control polling
    pending_events: WakerDeque<ToSwarm<Event, Infallible>>,
}

impl Behaviour {
    pub fn new(keypair: &identity::Keypair, bootstrap_peers: Vec<Multiaddr>) -> io::Result<Self> {
        Ok(Self {
            managed: managed::Behaviour::new(keypair)?,
            mdns_discovered: HashMap::new(),
            bootstrap_peers,
            connected_peers: HashMap::new(),
            retry_delay: Delay::new(RETRY_CONNECT_INTERVAL),
            pending_events: WakerDeque::new(),
        })
    }

    fn dial(&mut self, peer_id: PeerId, addr: Multiaddr) {
        self.pending_events.push_back(ToSwarm::Dial {
            opts: DialOpts::peer_id(peer_id).addresses(vec![addr]).build(),
        })
    }

    fn close_connection(&mut self, peer_id: PeerId, connection: ConnectionId) {
        // push front to make this IMMEDIATE
        self.pending_events.push_front(ToSwarm::CloseConnection {
            peer_id,
            connection: CloseConnection::One(connection),
        })
    }

    fn handle_mdns_discovered(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            if !self.is_connected(&p) {
                self.dial(p, ma.clone());
            }

            // get peer's multi-addresses or insert if missing
            let Some(mas) = self.mdns_discovered.get_mut(&p) else {
                self.mdns_discovered.insert(p, BTreeSet::from([ma]));
                continue;
            };

            // multiaddress should never already be present - else something has gone wrong
            let is_new_addr = mas.insert(ma);
            assert!(is_new_addr, "cannot discover a discovered peer");
        }
    }

    fn handle_mdns_expired(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            // at this point, we *must* have the peer
            let mas = self
                .mdns_discovered
                .get_mut(&p)
                .expect("nonexistent peer cannot expire");

            // at this point, we *must* have the multiaddress
            let was_present = mas.remove(&ma);
            assert!(was_present, "nonexistent multiaddress cannot expire");

            // if empty, remove the peer-id entirely
            if mas.is_empty() {
                self.mdns_discovered.remove(&p);
            }
        }
    }

    fn on_connection_established(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        *self.connected_peers.entry(peer_id).or_default() += 1;
        // send out connected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn on_connection_closed(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        if let Some(connection_count) = self.connected_peers.get_mut(&peer_id) {
            if *connection_count > 1 {
                *connection_count -= 1;
            } else {
                self.connected_peers.remove(&peer_id);
            }
        }
        // send out disconnected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn is_connected(&self, peer_id: &PeerId) -> bool {
        self.connected_peers.contains_key(peer_id)
    }

    fn should_redial_bootstrap_peer(&self, addr: &Multiaddr) -> bool {
        match addr.try_to_peer_id() {
            Some(peer_id) => !self.is_connected(&peer_id),
            None => true,
        }
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler =
        ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>;
    type ToSwarm = Event;

    // simply delegate to underlying mDNS behaviour

    delegate! {
        to self.managed {
            fn handle_pending_inbound_connection(&mut self, connection_id: ConnectionId, local_addr: &Multiaddr, remote_addr: &Multiaddr) -> Result<(), ConnectionDenied>;
            fn handle_pending_outbound_connection(&mut self, connection_id: ConnectionId, maybe_peer: Option<PeerId>, addresses: &[Multiaddr], effective_role: Endpoint) -> Result<Vec<Multiaddr>, ConnectionDenied>;
        }
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_inbound_connection(
                connection_id,
                peer,
                local_addr,
                remote_addr,
            )?,
        ))
    }

    #[allow(clippy::needless_question_mark)]
    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_outbound_connection(
                connection_id,
                peer,
                addr,
                role_override,
                port_use,
            )?,
        ))
    }

    fn on_connection_handler_event(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {
            Either::Left(ev) => libp2p::core::util::unreachable(ev),
            Either::Right(ev) => {
                self.managed
                    .on_connection_handler_event(peer_id, connection_id, ev)
            }
        }
    }

    // hook into these methods to drive behavior

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.managed.on_swarm_event(event); // let mDNS handle swarm events

        // handle swarm events to update internal state:
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection established event which is filtered correctly
                    self.on_connection_established(peer_id, connection_id, ip, port)
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection closed event which is filtered correctly
                    self.on_connection_closed(peer_id, connection_id, ip, port)
                }
            }

            // since we are running TCP/IP transport layer, we are assuming that
            // no address changes can occur, hence encountering one is a fatal error
            FromSwarm::AddressChange(a) => {
                unreachable!("unhandlable: address change encountered: {:?}", a)
            }
            _ => {}
        }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        // delegate to managed behaviors for any behaviors they need to perform
        match self.managed.poll(cx) {
            Poll::Ready(ToSwarm::GenerateEvent(e)) => {
                match e {
                    // handle discovered and expired events from mDNS
                    managed::BehaviourEvent::Mdns(e) => match e.clone() {
                        mdns::Event::Discovered(peers) => {
                            self.handle_mdns_discovered(peers);
                        }
                        mdns::Event::Expired(peers) => {
                            self.handle_mdns_expired(peers);
                        }
                    },

                    // handle ping events => if error then disconnect
                    managed::BehaviourEvent::Ping(e) => {
                        if let Err(_) = e.result {
                            self.close_connection(e.peer, e.connection.clone())
                        }
                    }
                }

                // since we just consumed an event, we should immediately wake just in case
                // there are more events to come where that came from
                cx.waker().wake_by_ref();
            }

            // forward any other mDNS event to the swarm or its connection handler(s)
            Poll::Ready(e) => {
                return Poll::Ready(
                    e.map_out(|_| unreachable!("events returning to swarm already handled"))
                        .map_in(Either::Right),
                );
            }

            Poll::Pending => {}
        }

        // retry connecting to all mDNS peers periodically (fails safely if already connected)
        if self.retry_delay.poll(cx).is_ready() {
            for (p, mas) in self.mdns_discovered.clone() {
                if self.is_connected(&p) {
                    continue;
                }
                for ma in mas {
                    self.dial(p, ma)
                }
            }
            // dial bootstrap peers (for environments where mDNS is unavailable)
            for addr in &self.bootstrap_peers {
                if !self.should_redial_bootstrap_peer(addr) {
                    continue;
                }
                self.pending_events.push_back(ToSwarm::Dial {
                    opts: DialOpts::unknown_peer_id().address(addr.clone()).build(),
                })
            }
            self.retry_delay.reset(RETRY_CONNECT_INTERVAL) // reset timeout
        }

        // send out any pending events from our own service
        if let Some(e) = self.pending_events.pop_front(cx) {
            return Poll::Ready(e.map_in(Either::Left));
        }

        // wait for pending events
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::Behaviour;
    use libp2p::{Multiaddr, PeerId, identity};

    #[tokio::test]
    async fn bootstrap_peer_with_active_connection_is_not_redialed() {
        let keypair = identity::Keypair::generate_ed25519();
        let connected_peer = identity::Keypair::generate_ed25519().public().to_peer_id();
        let addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/52416/p2p/{connected_peer}")
            .parse()
            .expect("valid multiaddr");

        let mut behaviour = Behaviour::new(&keypair, vec![addr.clone()]).expect("behaviour");
        behaviour.connected_peers.insert(connected_peer, 1);

        assert!(!behaviour.should_redial_bootstrap_peer(&addr));
    }

    #[tokio::test]
    async fn bootstrap_peer_without_active_connection_is_redialed() {
        let keypair = identity::Keypair::generate_ed25519();
        let target_peer = identity::Keypair::generate_ed25519().public().to_peer_id();
        let addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/52416/p2p/{target_peer}")
            .parse()
            .expect("valid multiaddr");

        let behaviour = Behaviour::new(&keypair, vec![addr.clone()]).expect("behaviour");

        assert!(behaviour.should_redial_bootstrap_peer(&addr));
    }

    #[tokio::test]
    async fn bootstrap_addr_without_peer_id_still_redials() {
        let keypair = identity::Keypair::generate_ed25519();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/52416".parse().expect("valid multiaddr");

        let behaviour = Behaviour::new(&keypair, vec![addr.clone()]).expect("behaviour");

        assert!(behaviour.should_redial_bootstrap_peer(&addr));
    }

    #[tokio::test]
    async fn connection_count_tracks_multiple_connections_per_peer() {
        let keypair = identity::Keypair::generate_ed25519();
        let peer_id: PeerId = identity::Keypair::generate_ed25519().public().to_peer_id();
        let mut behaviour = Behaviour::new(&keypair, vec![]).expect("behaviour");

        behaviour.connected_peers.insert(peer_id, 2);
        behaviour.on_connection_closed(
            peer_id,
            libp2p::swarm::ConnectionId::new_unchecked(1),
            "127.0.0.1".parse().expect("ip"),
            52416,
        );
        assert!(behaviour.is_connected(&peer_id));

        behaviour.on_connection_closed(
            peer_id,
            libp2p::swarm::ConnectionId::new_unchecked(2),
            "127.0.0.1".parse().expect("ip"),
            52416,
        );
        assert!(!behaviour.is_connected(&peer_id));
    }
}
