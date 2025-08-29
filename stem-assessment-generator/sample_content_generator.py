#!/usr/bin/env python3
"""
Sample PDF Generator for Testing
Creates realistic blockchain PDFs for testing the assessment generator
"""

from pathlib import Path
import os

class SampleContentGenerator:
    """Generate realistic blockchain content for testing"""
    
    def __init__(self):
        self.content_templates = {
            'basic_blockchain': self.generate_basic_blockchain_content(),
            'smart_contracts': self.generate_smart_contracts_content(),
            'advanced_topics': self.generate_advanced_topics_content()
        }
    
    def generate_basic_blockchain_content(self) -> str:
        """Generate basic blockchain educational content"""
        return """
Blockchain Technology Fundamentals

Chapter 1: Introduction to Blockchain

Blockchain is a revolutionary distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. By design, a blockchain is resistant to modification of data.

The key characteristics of blockchain technology include:

1. Decentralization: No single point of control or failure
2. Immutability: Once data is recorded, it cannot be easily changed
3. Transparency: All transactions are visible to network participants
4. Security: Cryptographic hashing protects data integrity

Chapter 2: Consensus Mechanisms

Consensus mechanisms are fundamental protocols that ensure all nodes in a blockchain network agree on the validity of transactions and the current state of the ledger. Without consensus, a distributed network would not be able to function reliably.

Proof of Work (PoW):
Proof of Work is the original consensus mechanism used by Bitcoin. In PoW, miners compete to solve computationally intensive mathematical puzzles to validate transactions and create new blocks. The first miner to solve the puzzle broadcasts the solution to the network, and other nodes verify the solution. If valid, the new block is added to the blockchain, and the miner receives a reward.

Advantages of PoW:
- High security due to computational requirements
- Proven track record with Bitcoin
- True decentralization

Disadvantages of PoW:
- High energy consumption
- Slower transaction processing
- Potential for mining centralization

Proof of Stake (PoS):
Proof of Stake is an alternative consensus mechanism where validators are chosen to create new blocks based on their stake in the network rather than computational power. Validators must lock up a certain amount of cryptocurrency as collateral.

Advantages of PoS:
- Energy efficient compared to PoW
- Faster transaction processing
- Lower barriers to participation

Disadvantages of PoS:
- Potential for wealth concentration
- "Nothing at stake" problem
- Less proven than PoW

Chapter 3: Mining and Validation

Mining is the process by which new transactions are verified and added to the blockchain. Despite the name, mining doesn't create cryptocurrency out of thin air - it validates transactions and maintains network security.

The mining process involves:
1. Transaction collection from the mempool
2. Verification of transaction validity
3. Creating a candidate block
4. Solving the proof-of-work puzzle
5. Broadcasting the solution to the network
6. Receiving confirmation from other nodes

Miners play several crucial roles:
- Transaction validation
- Network security maintenance
- New block creation
- Preventing double-spending attacks

Chapter 4: Cryptocurrency Wallets

A cryptocurrency wallet is a digital tool that allows users to store, send, and receive cryptocurrencies. Contrary to popular belief, wallets don't actually store cryptocurrency - they store the private keys that provide access to cryptocurrency addresses on the blockchain.

Types of wallets:

Hot Wallets (Online):
- Web wallets
- Mobile wallets  
- Desktop wallets
- Exchange wallets

Advantages: Convenient, easy access
Disadvantages: More vulnerable to hacking

Cold Wallets (Offline):
- Hardware wallets
- Paper wallets
- Air-gapped devices

Advantages: Higher security, immune to online attacks
Disadvantages: Less convenient for frequent transactions

Key Components:
- Public Key: Used to receive cryptocurrency (like an email address)
- Private Key: Used to spend cryptocurrency (like a password)
- Wallet Address: Derived from public key, used for transactions

Security Best Practices:
1. Never share your private key
2. Use hardware wallets for large amounts
3. Enable two-factor authentication
4. Keep backup copies of seed phrases
5. Verify addresses before sending transactions

Chapter 5: Cryptographic Hashing

Cryptographic hash functions are one-way mathematical functions that convert input data of any size into a fixed-size output called a hash or digest. Bitcoin and most blockchains use the SHA-256 hash function.

Properties of cryptographic hash functions:
1. Deterministic: Same input always produces the same output
2. Fixed output size: SHA-256 always produces 256-bit output
3. Avalanche effect: Small input changes cause dramatic output changes
4. One-way: Easy to compute hash from input, extremely difficult to reverse
5. Collision resistant: Very difficult to find two inputs with same hash

Applications in blockchain:
- Block linking: Each block contains the hash of the previous block
- Transaction verification: Merkle trees use hashes to efficiently verify transactions
- Mining: Proof-of-work requires finding specific hash patterns
- Digital signatures: Hashing is part of the signing process

The importance of hashing cannot be overstated - it provides the cryptographic foundation that makes blockchain technology secure and tamper-evident.
"""

    def generate_smart_contracts_content(self) -> str:
        """Generate smart contracts educational content"""
        return """
Smart Contracts and Ethereum

Chapter 1: Introduction to Smart Contracts

Smart contracts are self-executing contracts with the terms of the agreement directly written into code. They automatically execute when predetermined conditions are met, eliminating the need for intermediaries and reducing the potential for disputes.

The concept was first proposed by computer scientist Nick Szabo in 1994, long before blockchain technology existed. However, it wasn't until the creation of Ethereum that smart contracts became practically implementable.

Key characteristics of smart contracts:
1. Autonomous execution
2. Transparent and verifiable code
3. Immutable once deployed
4. Cost-effective (no intermediaries)
5. Trustless operation

Chapter 2: Ethereum Platform

Ethereum is a decentralized platform that enables developers to build and deploy smart contracts and decentralized applications (DApps). Unlike Bitcoin, which is primarily a digital currency, Ethereum is designed as a programmable blockchain.

Ethereum Virtual Machine (EVM):
The EVM is the runtime environment for smart contracts in Ethereum. It's a quasi-Turing complete virtual machine that executes bytecode compiled from high-level languages like Solidity.

Key features of the EVM:
- Deterministic execution
- Isolated environment
- Gas-based execution model
- State management

Ethereum Accounts:
1. Externally Owned Accounts (EOAs): Controlled by private keys
2. Contract Accounts: Controlled by smart contract code

Chapter 3: Gas and Transaction Fees

Gas is a unit of measurement for the computational effort required to execute operations on the Ethereum network. Every operation in the EVM consumes a specific amount of gas.

Gas serves multiple purposes:
1. Prevents infinite loops and spam
2. Compensates miners for computational resources
3. Prioritizes transactions during network congestion

Gas concepts:
- Gas Limit: Maximum gas a transaction can consume
- Gas Price: Price per unit of gas (in Gwei)
- Gas Used: Actual gas consumed by the transaction
- Transaction Fee = Gas Used × Gas Price

Gas optimization strategies:
1. Efficient code writing
2. Proper data structure selection
3. Minimizing external calls
4. Using events instead of storage for logs

Chapter 4: Solidity Programming Language

Solidity is a statically-typed programming language designed for developing smart contracts that run on the Ethereum Virtual Machine. It's influenced by C++, Python, and JavaScript.

Basic Solidity structure:
```solidity
pragma solidity ^0.8.0;

contract MyContract {
    // State variables
    uint256 public value;
    address public owner;
    
    // Events
    event ValueChanged(uint256 newValue);
    
    // Constructor
    constructor() {
        owner = msg.sender;
    }
    
    // Functions
    function setValue(uint256 _value) public {
        require(msg.sender == owner, "Only owner can set value");
        value = _value;
        emit ValueChanged(_value);
    }
}
```

Solidity features:
- Strong typing
- Inheritance support
- Libraries and interfaces
- Modifier system
- Event logging

Chapter 5: Smart Contract Security

Smart contract security is crucial because deployed contracts are immutable and often handle valuable assets. Common vulnerabilities include:

Reentrancy Attacks:
Occurs when a contract calls an external contract before updating its own state, allowing the external contract to call back and potentially drain funds.

Prevention:
- Use checks-effects-interactions pattern
- Implement reentrancy guards
- Update state before external calls

Integer Overflow/Underflow:
Mathematical operations that exceed the maximum or minimum values for integer types.

Prevention:
- Use SafeMath library (Solidity < 0.8.0)
- Use built-in overflow protection (Solidity ≥ 0.8.0)
- Validate input parameters

Access Control Issues:
Inadequate permission systems allowing unauthorized access to critical functions.

Prevention:
- Implement proper role-based access control
- Use OpenZeppelin's access control contracts
- Regular security audits

Chapter 6: Decentralized Applications (DApps)

DApps are applications that run on a decentralized network, typically using smart contracts as their backend. They combine smart contracts with user interfaces to create fully functional applications.

DApp architecture:
1. Frontend: Web interface (HTML, CSS, JavaScript)
2. Backend: Smart contracts on blockchain
3. Storage: IPFS or other decentralized storage
4. Communication: Web3 libraries

Advantages of DApps:
- Censorship resistant
- No single point of failure
- Transparent operation
- Global accessibility

Challenges:
- Scalability limitations
- High transaction costs
- Complex user experience
- Regulatory uncertainty

Popular DApp categories:
1. Decentralized Finance (DeFi)
2. Non-Fungible Tokens (NFTs)
3. Gaming and collectibles
4. Governance and voting
5. Social networks

Chapter 7: Future of Smart Contracts

The future of smart contracts includes several exciting developments:

Layer 2 Solutions:
- Optimistic Rollups
- Zero-Knowledge Rollups
- State Channels
- Sidechains

These solutions aim to improve scalability while maintaining security.

Interoperability:
Cross-chain protocols enabling smart contracts to interact across different blockchain networks.

Formal Verification:
Mathematical proofs that smart contracts behave correctly according to specifications.

Oracles:
Reliable data feeds that connect smart contracts to real-world information.

The smart contract ecosystem continues to evolve rapidly, with new tools, frameworks, and best practices emerging regularly.
"""

    def generate_advanced_topics_content(self) -> str:
        """Generate advanced blockchain topics content"""
        return """
Advanced Blockchain Technologies

Chapter 1: Blockchain Scalability

Scalability is one of the most significant challenges facing blockchain technology today. The blockchain trilemma, coined by Ethereum creator Vitalik Buterin, describes the difficulty of achieving decentralization, security, and scalability simultaneously.

The Blockchain Trilemma:
- Decentralization: No single entity controls the network
- Security: Network is resistant to attacks and failures
- Scalability: Network can process many transactions quickly

Traditional blockchains typically excel at two of these properties but struggle with the third.

Scalability Metrics:
- Transactions Per Second (TPS)
- Transaction Confirmation Time
- Network Throughput
- Storage Requirements

Bitcoin processes ~7 TPS, Ethereum ~15 TPS, while traditional payment processors like Visa handle ~65,000 TPS.

Chapter 2: Layer 2 Solutions

Layer 2 solutions are protocols built on top of existing blockchains to improve scalability without compromising the underlying blockchain's security.

State Channels:
State channels allow participants to conduct multiple transactions off-chain, only settling the final state on the main blockchain.

Benefits:
- Instant transactions
- Lower fees
- Private transactions
- Micropayment support

Examples: Lightning Network (Bitcoin), Raiden Network (Ethereum)

Sidechains:
Sidechains are separate blockchains that run parallel to the main blockchain, with a two-way peg mechanism for asset transfer.

Characteristics:
- Independent consensus mechanism
- Separate security model
- Asset bridging capability
- Specialized functionality

Examples: Polygon, Liquid Network

Optimistic Rollups:
Optimistic rollups execute transactions off-chain and periodically submit transaction data to the main chain, assuming transactions are valid unless challenged.

Features:
- Higher throughput
- EVM compatibility
- Fraud proof mechanism
- 7-day withdrawal period

Examples: Arbitrum, Optimism

Zero-Knowledge Rollups:
ZK-rollups use cryptographic proofs to verify transaction validity without revealing transaction details.

Advantages:
- Instant finality
- Privacy preservation
- High security
- Scalability improvement

Examples: zkSync, StarkEx

Chapter 3: Sharding

Sharding is a method of partitioning the blockchain network into smaller, more manageable pieces called shards, each capable of processing transactions independently.

Types of Sharding:

Network Sharding:
Dividing the network nodes into different groups, each responsible for processing specific transactions.

Transaction Sharding:
Distributing different transactions across multiple shards for parallel processing.

State Sharding:
Partitioning the blockchain state across multiple shards, with each shard maintaining a portion of the overall state.

Challenges:
- Cross-shard communication
- Security maintenance
- Load balancing
- Atomic transactions across shards

Ethereum 2.0 Sharding:
Ethereum's transition to proof-of-stake includes plans for 64 shard chains, significantly increasing the network's capacity.

Chapter 4: Interoperability and Cross-Chain Protocols

Blockchain interoperability enables different blockchain networks to communicate and share information seamlessly.

Interoperability Approaches:

Atomic Swaps:
Peer-to-peer exchange of cryptocurrencies from different blockchains without trusted intermediaries.

Requirements:
- Hash Time-Locked Contracts (HTLCs)
- Compatible cryptographic functions
- Network consensus on both chains

Bridge Protocols:
Smart contracts that lock assets on one blockchain and mint equivalent assets on another.

Types:
- Centralized bridges (trusted third parties)
- Decentralized bridges (smart contract-based)
- Federated bridges (multi-signature schemes)

Relay Chains:
Specialized blockchains designed to connect and coordinate multiple blockchain networks.

Examples: Polkadot, Cosmos

Cross-Chain Communication Protocols:
- Inter-Blockchain Communication (IBC)
- Cross-Chain Bridge Protocol (CCIP)
- Wormhole Protocol

Chapter 5: Consensus Algorithm Innovations

Beyond traditional PoW and PoS, several innovative consensus mechanisms have emerged:

Delegated Proof of Stake (DPoS):
Token holders vote for delegates who validate transactions and maintain the network.

Advantages:
- High throughput
- Energy efficiency
- Democratic participation

Examples: EOS, Tron, Cardano

Proof of Authority (PoA):
Pre-approved validators are responsible for validating transactions and creating blocks.

Use cases:
- Private networks
- Consortium blockchains
- Testing environments

Proof of Space (PoSpace):
Validators prove they're dedicating disk space to the network rather than computational power.

Benefits:
- Energy efficient
- Accessible to more participants
- Environmentally friendly

Example: Chia Network

Practical Byzantine Fault Tolerance (pBFT):
Designed for permissioned networks where validators are known and communication is reliable.

Characteristics:
- Immediate finality
- High performance
- Limited scalability

Chapter 6: Privacy-Preserving Technologies

Zero-Knowledge Proofs:
Cryptographic methods that allow one party to prove knowledge of information without revealing the information itself.

Types:
- zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge)
- zk-STARKs (Zero-Knowledge Scalable Transparent Arguments of Knowledge)

Applications:
- Private transactions
- Identity verification
- Regulatory compliance
- Voting systems

Ring Signatures:
Digital signatures that can be performed by any member of a group of users, without revealing which member actually signed.

Benefits:
- Anonymity protection
- Plausible deniability
- Group authentication

Example: Monero cryptocurrency

Homomorphic Encryption:
Encryption that allows computations to be performed on ciphertexts without decrypting them.

Use cases:
- Private smart contracts
- Confidential data analysis
- Secure multi-party computation

Chapter 7: Decentralized Autonomous Organizations (DAOs)

DAOs are organizations governed by smart contracts and operated by their members without traditional management structures.

DAO Components:
- Governance tokens
- Voting mechanisms
- Proposal systems
- Treasury management
- Execution protocols

Governance Models:
- Token-based voting
- Quadratic voting
- Conviction voting
- Delegated voting

Challenges:
- Legal recognition
- Governance attacks
- Voter apathy
- Technical complexity

Examples: MakerDAO, Compound, Uniswap

Chapter 8: Future Trends and Developments

Quantum Resistance:
As quantum computers advance, blockchain networks must adopt quantum-resistant cryptographic algorithms.

Post-quantum cryptography:
- Lattice-based cryptography
- Hash-based signatures
- Code-based cryptography
- Multivariate cryptography

Central Bank Digital Currencies (CBDCs):
Government-issued digital currencies built on blockchain technology.

Features:
- Government backing
- Regulatory compliance
- Privacy controls
- Programmable money

Web3 Integration:
The evolution toward a decentralized internet built on blockchain infrastructure.

Components:
- Decentralized storage (IPFS, Filecoin)
- Decentralized computing (Ethereum, Solana)
- Decentralized identity (DIDs)
- Decentralized naming (ENS)

The blockchain ecosystem continues to evolve rapidly, with new innovations addressing current limitations and opening new possibilities for decentralized systems.
"""

    def create_sample_pdfs(self, output_dir: str = "test_data"):
        """Create sample PDF files for testing"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📝 Creating sample content in {output_path}")
        
        # For now, create text files (in production, would use reportlab to create actual PDFs)
        for name, content in self.content_templates.items():
            file_path = output_path / f"{name}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Created: {file_path} ({len(content.split())} words)")
        
        # Create a simple README
        readme_content = """
# Test Data Directory

This directory contains sample blockchain educational content for testing the STEM Assessment Generator.

## Files:
- basic_blockchain.txt: Fundamentals of blockchain technology
- smart_contracts.txt: Smart contracts and Ethereum
- advanced_topics.txt: Advanced blockchain concepts

## Usage:
These files can be used to test PDF processing and question generation.
For actual PDF testing, convert these to PDF format using any text-to-PDF tool.

## Note:
In a production environment, these would be actual PDF files.
The current implementation creates text files for testing purposes.
"""
        
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Created: {readme_path}")
        print(f"\n📋 Test data ready in {output_path}")
        
        return output_path

def main():
    """Generate sample content for testing"""
    generator = SampleContentGenerator()
    
    print("🚀 Sample Content Generator for STEM Assessment Testing")
    print("=" * 60)
    
    # Create sample files
    output_dir = generator.create_sample_pdfs()
    
    print(f"\n💡 Next steps:")
    print(f"1. Convert text files to PDF format if needed")
    print(f"2. Run: python test_full_pipeline.py")
    print(f"3. Use files in {output_dir} for testing")
    print(f"4. Check quality with: python quality_validator.py")

if __name__ == "__main__":
    main()
