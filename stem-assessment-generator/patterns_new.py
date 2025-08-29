"""
Comprehensive Blockchain Misconception Patterns for Distractor Generation

This module provides sophisticated patterns to generate realistic but incorrect 
distractors for blockchain-related multiple choice questions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re
import random

class DifficultyLevel(str, Enum):
    """Difficulty levels for patterns"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ConceptCategory(str, Enum):
    """Blockchain concept categories"""
    CONSENSUS = "consensus"
    CRYPTOGRAPHY = "cryptography"
    SMART_CONTRACTS = "smart_contracts"
    MINING = "mining"
    WALLETS = "wallets"
    TRANSACTIONS = "transactions"
    DECENTRALIZATION = "decentralization"
    SECURITY = "security"

# =============================================================================
# Base Pattern Classes
# =============================================================================

class MisconceptionPattern(ABC):
    """Base class for all misconception patterns"""
    
    def __init__(self, name: str, difficulty_range: List[DifficultyLevel], 
                 closeness_range: Tuple[int, int], concepts: List[ConceptCategory]):
        self.name = name
        self.difficulty_range = difficulty_range
        self.closeness_range = closeness_range  # 1-10 scale
        self.concepts = concepts
        self.description = ""
    
    @abstractmethod
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        """Apply the pattern to generate a distractor"""
        pass
    
    def is_applicable(self, text: str, difficulty: DifficultyLevel) -> bool:
        """Check if pattern can be applied to the given text and difficulty"""
        return (difficulty in self.difficulty_range and 
                self._has_applicable_concepts(text))
    
    def _has_applicable_concepts(self, text: str) -> bool:
        """Check if text contains concepts this pattern applies to"""
        text_lower = text.lower()
        concept_keywords = {
            ConceptCategory.CONSENSUS: ['consensus', 'proof of work', 'proof of stake', 'mining', 'validation'],
            ConceptCategory.CRYPTOGRAPHY: ['hash', 'encryption', 'key', 'signature', 'cryptographic'],
            ConceptCategory.SMART_CONTRACTS: ['smart contract', 'ethereum', 'solidity', 'dapp', 'gas'],
            ConceptCategory.MINING: ['mining', 'miner', 'proof of work', 'hash rate', 'block reward'],
            ConceptCategory.WALLETS: ['wallet', 'private key', 'public key', 'address', 'seed'],
            ConceptCategory.TRANSACTIONS: ['transaction', 'transfer', 'input', 'output', 'utxo'],
            ConceptCategory.DECENTRALIZATION: ['decentralized', 'distributed', 'peer-to-peer', 'node'],
            ConceptCategory.SECURITY: ['security', 'attack', 'vulnerability', 'immutable', 'tamper']
        }
        
        for concept in self.concepts:
            keywords = concept_keywords.get(concept, [])
            if any(keyword in text_lower for keyword in keywords):
                return True
        return False

# =============================================================================
# Terminology Confusion Patterns
# =============================================================================

class WalletVsNodePattern(MisconceptionPattern):
    """Confuses wallet functionality with node functionality"""
    
    def __init__(self):
        super().__init__(
            name="wallet_vs_node",
            difficulty_range=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
            closeness_range=(5, 7),
            concepts=[ConceptCategory.WALLETS, ConceptCategory.DECENTRALIZATION]
        )
        self.description = "Confuses wallet storage vs node storage"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        # Wallet stores keys -> Node stores blockchain
        if 'wallet' in text_lower and ('key' in text_lower or 'private' in text_lower):
            return correct_answer.replace('private key', 'blockchain data').replace('keys', 'blockchain')
        
        # Node validates -> Wallet validates
        if 'node' in text_lower and 'validat' in text_lower:
            return correct_answer.replace('node', 'wallet').replace('Node', 'Wallet')
        
        return None

class MiningVsValidatingPattern(MisconceptionPattern):
    """Confuses mining with simple validation"""
    
    def __init__(self):
        super().__init__(
            name="mining_vs_validating",
            difficulty_range=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.MINING, ConceptCategory.CONSENSUS]
        )
        self.description = "Reduces mining to just creating coins"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'mining' in text_lower or 'miner' in text_lower:
            # Emphasize coin creation over validation
            if 'validat' in text_lower and 'transaction' in text_lower:
                return correct_answer.replace('validate transactions', 'generate new cryptocurrency')
            elif 'block' in text_lower and 'create' in text_lower:
                return correct_answer.replace('create new blocks', 'mine cryptocurrency rewards')
        
        return None

class PublicVsPrivateKeyPattern(MisconceptionPattern):
    """Reverses public and private key roles"""
    
    def __init__(self):
        super().__init__(
            name="public_vs_private_key",
            difficulty_range=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
            closeness_range=(5, 7),
            concepts=[ConceptCategory.CRYPTOGRAPHY, ConceptCategory.WALLETS]
        )
        self.description = "Swaps public and private key functions"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        # Swap public and private key roles
        swapped = correct_answer.replace('private key', '__TEMP__')
        swapped = swapped.replace('public key', 'private key')
        swapped = swapped.replace('__TEMP__', 'public key')
        
        # Also swap Public/Private
        swapped = swapped.replace('Private key', '__TEMP__')
        swapped = swapped.replace('Public key', 'Private key')
        swapped = swapped.replace('__TEMP__', 'Public key')
        
        return swapped if swapped != correct_answer else None

class HashVsEncryptionPattern(MisconceptionPattern):
    """Confuses hashing with encryption"""
    
    def __init__(self):
        super().__init__(
            name="hash_vs_encryption",
            difficulty_range=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.CRYPTOGRAPHY]
        )
        self.description = "Treats hashing as reversible like encryption"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'hash' in text_lower:
            # Make hashing sound reversible
            replacements = [
                ('one-way', 'two-way'),
                ('irreversible', 'reversible'),
                ('cannot be reversed', 'can be decrypted'),
                ('hash function', 'encryption algorithm')
            ]
            
            result = correct_answer
            for old, new in replacements:
                if old in text_lower:
                    result = result.replace(old, new)
            
            return result if result != correct_answer else None
        
        return None

# =============================================================================
# Conceptual Error Patterns
# =============================================================================

class BlockchainAsDatabasePattern(MisconceptionPattern):
    """Treats blockchain as a simple database"""
    
    def __init__(self):
        super().__init__(
            name="blockchain_as_database",
            difficulty_range=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
            closeness_range=(5, 7),
            concepts=[ConceptCategory.DECENTRALIZATION]
        )
        self.description = "Misses distributed and immutable aspects"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'blockchain' in text_lower:
            replacements = [
                ('distributed', 'centralized'),
                ('decentralized', 'traditional'),
                ('immutable', 'updateable'),
                ('consensus', 'administrator approval'),
                ('peer-to-peer', 'client-server')
            ]
            
            result = correct_answer
            for old, new in replacements:
                if old in text_lower:
                    result = result.replace(old, new)
            
            return result if result != correct_answer else None
        
        return None

class DecentralizationEqualsSpeedPattern(MisconceptionPattern):
    """Assumes decentralization makes things faster"""
    
    def __init__(self):
        super().__init__(
            name="decentralization_equals_speed",
            difficulty_range=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.DECENTRALIZATION, ConceptCategory.CONSENSUS]
        )
        self.description = "Incorrectly assumes decentralization improves speed"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'decentral' in text_lower:
            if 'slow' in text_lower or 'time' in text_lower:
                result = correct_answer.replace('slower', 'faster')
                result = result.replace('slow', 'fast')
                return result
            elif 'consensus' in text_lower:
                return correct_answer + ' making transactions much faster'
        
        return None

class ConsensusIsVotingPattern(MisconceptionPattern):
    """Oversimplifies consensus as simple voting"""
    
    def __init__(self):
        super().__init__(
            name="consensus_is_voting",
            difficulty_range=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.CONSENSUS]
        )
        self.description = "Reduces consensus protocols to simple voting"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'consensus' in text_lower:
            replacements = [
                ('consensus mechanism', 'voting system'),
                ('proof of work', 'majority vote'),
                ('proof of stake', 'stakeholder vote'),
                ('protocol', 'poll'),
                ('algorithm', 'ballot counting')
            ]
            
            result = correct_answer
            for old, new in replacements:
                if old in text_lower:
                    result = result.replace(old, new)
            
            return result if result != correct_answer else None
        
        return None

class ImmutableMeansUnhackablePattern(MisconceptionPattern):
    """Confuses immutability with security"""
    
    def __init__(self):
        super().__init__(
            name="immutable_means_unhackable",
            difficulty_range=[DifficultyLevel.HARD],
            closeness_range=(7, 9),
            concepts=[ConceptCategory.SECURITY]
        )
        self.description = "Confuses data permanence with system security"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'immutable' in text_lower or 'tamper' in text_lower:
            # Add false security claims
            if 'cannot be changed' in text_lower:
                return correct_answer + ', making it completely secure from all attacks'
            elif 'permanent' in text_lower:
                return correct_answer.replace('permanent', 'hack-proof')
        
        return None

# =============================================================================
# Technical Mixup Patterns
# =============================================================================

class PowVsPosPattern(MisconceptionPattern):
    """Swaps Proof of Work and Proof of Stake mechanisms"""
    
    def __init__(self):
        super().__init__(
            name="pow_vs_pos",
            difficulty_range=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
            closeness_range=(7, 9),
            concepts=[ConceptCategory.CONSENSUS, ConceptCategory.MINING]
        )
        self.description = "Swaps computational work with stake requirements"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        replacements = [
            ('proof of work', '__POS_TEMP__'),
            ('proof of stake', 'proof of work'),
            ('__POS_TEMP__', 'proof of stake'),
            ('computational power', 'token holdings'),
            ('mining difficulty', 'stake amount'),
            ('energy consumption', 'token lock-up'),
            ('hash rate', 'stake weight')
        ]
        
        result = correct_answer
        for old, new in replacements:
            result = result.replace(old, new)
        
        return result if result != correct_answer else None

class GasFeesPurposePattern(MisconceptionPattern):
    """Misunderstands gas fees as simple transaction fees"""
    
    def __init__(self):
        super().__init__(
            name="gas_fees_purpose",
            difficulty_range=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.SMART_CONTRACTS, ConceptCategory.TRANSACTIONS]
        )
        self.description = "Reduces gas to simple transaction fee"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'gas' in text_lower:
            replacements = [
                ('computational resources', 'transaction fees'),
                ('execution cost', 'network fee'),
                ('prevents infinite loops', 'rewards miners'),
                ('limits computation', 'covers bandwidth')
            ]
            
            result = correct_answer
            for old, new in replacements:
                if old in text_lower:
                    result = result.replace(old, new)
            
            return result if result != correct_answer else None
        
        return None

class SmartContractCapabilitiesPattern(MisconceptionPattern):
    """Overestimates smart contract capabilities"""
    
    def __init__(self):
        super().__init__(
            name="smart_contract_capabilities",
            difficulty_range=[DifficultyLevel.HARD],
            closeness_range=(7, 8),
            concepts=[ConceptCategory.SMART_CONTRACTS]
        )
        self.description = "Assumes contracts can access external data directly"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'smart contract' in text_lower:
            if 'oracle' in text_lower or 'external' in text_lower:
                return correct_answer.replace('oracle', 'built-in internet access')
            elif 'blockchain' in text_lower and 'data' in text_lower:
                return correct_answer + ' and can directly access any internet API'
        
        return None

class MerkleTreePurposePattern(MisconceptionPattern):
    """Provides wrong explanations for Merkle trees"""
    
    def __init__(self):
        super().__init__(
            name="merkle_tree_purpose",
            difficulty_range=[DifficultyLevel.HARD],
            closeness_range=(6, 8),
            concepts=[ConceptCategory.CRYPTOGRAPHY]
        )
        self.description = "Misunderstands Merkle tree verification purpose"
    
    def apply(self, correct_answer: str, context: Optional[str] = None) -> Optional[str]:
        text_lower = correct_answer.lower()
        
        if 'merkle' in text_lower or 'tree' in text_lower:
            wrong_purposes = [
                'stores user passwords securely',
                'manages transaction ordering',
                'enables faster mining',
                'provides data compression'
            ]
            
            if 'verif' in text_lower:
                return correct_answer.replace('verification', random.choice(wrong_purposes))
        
        return None

# =============================================================================
# Pattern Application Logic
# =============================================================================

class PatternAnalyzer:
    """Analyzes text and applies appropriate misconception patterns"""
    
    def __init__(self):
        self.patterns = [
            WalletVsNodePattern(),
            MiningVsValidatingPattern(),
            PublicVsPrivateKeyPattern(),
            HashVsEncryptionPattern(),
            BlockchainAsDatabasePattern(),
            DecentralizationEqualsSpeedPattern(),
            ConsensusIsVotingPattern(),
            ImmutableMeansUnhackablePattern(),
            PowVsPosPattern(),
            GasFeesPurposePattern(),
            SmartContractCapabilitiesPattern(),
            MerkleTreePurposePattern()
        ]
    
    def analyze_correct_answer(self, text: str) -> List[ConceptCategory]:
        """Identify key blockchain concepts in the text"""
        concepts = []
        text_lower = text.lower()
        
        concept_keywords = {
            ConceptCategory.CONSENSUS: ['consensus', 'proof of work', 'proof of stake', 'mining', 'validation'],
            ConceptCategory.CRYPTOGRAPHY: ['hash', 'encryption', 'key', 'signature', 'cryptographic', 'merkle'],
            ConceptCategory.SMART_CONTRACTS: ['smart contract', 'ethereum', 'solidity', 'dapp', 'gas'],
            ConceptCategory.MINING: ['mining', 'miner', 'proof of work', 'hash rate', 'block reward'],
            ConceptCategory.WALLETS: ['wallet', 'private key', 'public key', 'address', 'seed'],
            ConceptCategory.TRANSACTIONS: ['transaction', 'transfer', 'input', 'output', 'utxo'],
            ConceptCategory.DECENTRALIZATION: ['decentralized', 'distributed', 'peer-to-peer', 'node'],
            ConceptCategory.SECURITY: ['security', 'attack', 'vulnerability', 'immutable', 'tamper']
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def select_applicable_patterns(self, text: str, difficulty: DifficultyLevel) -> List[MisconceptionPattern]:
        """Select patterns applicable to the text and difficulty"""
        applicable = []
        
        for pattern in self.patterns:
            if pattern.is_applicable(text, difficulty):
                applicable.append(pattern)
        
        return applicable
    
    def generate_distractors(self, correct_answer: str, difficulty: DifficultyLevel, 
                           num_needed: int = 3, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate distractors using applicable patterns"""
        applicable_patterns = self.select_applicable_patterns(correct_answer, difficulty)
        distractors = []
        
        # Shuffle patterns for variety
        random.shuffle(applicable_patterns)
        
        for pattern in applicable_patterns:
            if len(distractors) >= num_needed:
                break
            
            distractor_text = pattern.apply(correct_answer, context)
            if distractor_text and self.validate_distractor_quality(distractor_text, correct_answer):
                distractors.append({
                    'text': distractor_text,
                    'pattern': pattern.name,
                    'closeness': random.randint(*pattern.closeness_range),
                    'difficulty': difficulty.value,
                    'concepts': [c.value for c in pattern.concepts]
                })
        
        return distractors[:num_needed]
    
    def validate_distractor_quality(self, distractor: str, correct_answer: str) -> bool:
        """Validate that the distractor is different but plausible"""
        # Must be different from correct answer
        if distractor.strip().lower() == correct_answer.strip().lower():
            return False
        
        # Must have reasonable length (not too short or too long)
        length_ratio = len(distractor) / len(correct_answer)
        if length_ratio < 0.5 or length_ratio > 2.0:
            return False
        
        # Must contain some blockchain-related terms
        blockchain_terms = ['blockchain', 'bitcoin', 'crypto', 'hash', 'key', 'transaction', 
                          'block', 'mining', 'wallet', 'consensus', 'node', 'contract']
        
        distractor_lower = distractor.lower()
        has_blockchain_term = any(term in distractor_lower for term in blockchain_terms)
        
        return has_blockchain_term

# =============================================================================
# Pattern Registry Export
# =============================================================================

def create_pattern_registry() -> Dict[str, Any]:
    """Create a registry of all patterns organized by difficulty and concept"""
    analyzer = PatternAnalyzer()
    
    registry = {
        "difficulty": {
            "EASY": [],
            "MEDIUM": [],
            "HARD": []
        },
        "concept": {
            "consensus": [],
            "cryptography": [],
            "smart_contracts": [],
            "mining": [],
            "wallets": [],
            "transactions": [],
            "decentralization": [],
            "security": []
        },
        "patterns": analyzer.patterns
    }
    
    # Organize patterns by difficulty
    for pattern in analyzer.patterns:
        for difficulty in pattern.difficulty_range:
            registry["difficulty"][difficulty.value.upper()].append(pattern)
    
    # Organize patterns by concept
    for pattern in analyzer.patterns:
        for concept in pattern.concepts:
            registry["concept"][concept.value].append(pattern)
    
    return registry

# Export the main components
PATTERN_REGISTRY = create_pattern_registry()
pattern_analyzer = PatternAnalyzer()

# =============================================================================
# Convenience Functions
# =============================================================================

def generate_blockchain_distractors(correct_answer: str, difficulty: str = "medium", 
                                   num_distractors: int = 3) -> List[str]:
    """Convenience function to generate distractors for a correct answer"""
    difficulty_level = DifficultyLevel(difficulty.lower())
    distractors = pattern_analyzer.generate_distractors(
        correct_answer, difficulty_level, num_distractors
    )
    return [d['text'] for d in distractors]

def analyze_answer_concepts(text: str) -> List[str]:
    """Convenience function to analyze concepts in text"""
    concepts = pattern_analyzer.analyze_correct_answer(text)
    return [c.value for c in concepts]

def get_patterns_for_difficulty(difficulty: str) -> List[str]:
    """Get pattern names for a specific difficulty"""
    patterns = PATTERN_REGISTRY["difficulty"].get(difficulty.upper(), [])
    return [p.name for p in patterns]

def get_patterns_for_concept(concept: str) -> List[str]:
    """Get pattern names for a specific concept"""
    patterns = PATTERN_REGISTRY["concept"].get(concept.lower(), [])
    return [p.name for p in patterns]

# Example usage and testing
if __name__ == "__main__":
    # Test the pattern system
    test_answers = [
        "Miners validate transactions and create new blocks through proof of work",
        "A private key is used to sign transactions while the public key verifies them",
        "Smart contracts execute automatically when conditions are met",
        "Blockchain provides immutable and decentralized transaction records",
        "Gas fees limit computational resources in smart contract execution"
    ]
    
    print("=== Blockchain Misconception Pattern Testing ===\n")
    
    for i, answer in enumerate(test_answers, 1):
        print(f"Test {i}: {answer}")
        print("-" * 60)
        
        # Analyze concepts
        concepts = analyze_answer_concepts(answer)
        print(f"Detected concepts: {', '.join(concepts)}")
        
        # Generate distractors for each difficulty
        for difficulty in ["easy", "medium", "hard"]:
            distractors = generate_blockchain_distractors(answer, difficulty, 2)
            if distractors:
                print(f"\n{difficulty.upper()} distractors:")
                for j, distractor in enumerate(distractors, 1):
                    print(f"  {j}. {distractor}")
        
        print("\n" + "="*80 + "\n")
