<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Bootstrap peers

Bootstrap peers provide the first contact points for the network. They help a new node discover other peers and join the current network.

Bootstrap peers are not the blockchain, not the owner of the network, and not a permanent proxy for model traffic.
Peer payloads carry a separate `genesis_hash`, so a node that uses a different genesis block is rejected even if it advertises the same `mainnet` chain name.

## How bootstrap peers are used

1. A node starts with a configured list of known peers.
2. The node contacts one or more bootstrap peers.
3. The node receives peer information and network status.
4. Runtime and transport checks decide which peers are actually usable.
5. Model data moves through direct PC-to-PC paths when possible, or through relay paths only when needed for connectivity.

## Configuration model

Bootstrap peers should be treated as an array/list. This allows operators to add more public contact points without creating a new genesis or a competing network.

Recommended properties:

- multiple entries instead of a single hardcoded peer;
- stable peer IDs and addresses;
- no private keys or seed phrases in peer lists;
- clear separation between discovery configuration and chain state;
- documented changes through pull requests.

## Validator discovery

Validators also need initial connectivity. They use the same principle: bootstrap peers help nodes find the network, while validator set and settlement rules come from chain state and project configuration.

Adding a bootstrap peer does not automatically make it a validator. A validator must run validator mode, hold the required operational wallet/key material, and participate in settlement according to the network rules.

## Safety rules

- Do not regenerate genesis just to add a bootstrap peer.
- Do not publish VPS passwords, private keys, seed phrases, or API tokens.
- Do not rely on one bootstrap address as the only network entry point.
- Keep peer additions reviewable and easy to audit.
