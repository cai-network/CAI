// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!
pub mod discovery;
pub mod swarm;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use extend::ext;
    use libp2p::{Multiaddr, PeerId};
    use libp2p::multiaddr::Protocol;
    use std::net::IpAddr;

    #[ext(pub, name = MultiaddrExt)]
    impl Multiaddr {
        /// If the multiaddress corresponds to a TCP address, extracts it
        fn try_to_tcp_addr(&self) -> Option<(IpAddr, u16)> {
            let mut ps = self.into_iter();
            let ip = if let Some(p) = ps.next() {
                match p {
                    Protocol::Ip4(ip) => IpAddr::V4(ip),
                    Protocol::Ip6(ip) => IpAddr::V6(ip),
                    _ => return None,
                }
            } else {
                return None;
            };
            let Some(Protocol::Tcp(port)) = ps.next() else {
                return None;
            };
            Some((ip, port))
        }

        /// If the multiaddress carries a `/p2p/<peer-id>` component, extracts it.
        fn try_to_peer_id(&self) -> Option<PeerId> {
            self.into_iter().find_map(|protocol| match protocol {
                Protocol::P2p(peer_id) => Some(peer_id),
                _ => None,
            })
        }
    }
}
