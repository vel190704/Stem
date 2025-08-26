"""
Misconception patterns for generating realistic distractors
"""
from typing import Dict, List
from models import MisconceptionPattern

# Common STEM misconception patterns
MISCONCEPTION_PATTERNS = {
    "physics": {
        "force_and_motion": MisconceptionPattern(
            name="Force and Motion Confusion",
            description="Students confuse concepts of force, velocity, and acceleration",
            examples=[
                "Thinking force is needed to maintain constant velocity",
                "Confusing acceleration with velocity",
                "Believing heavier objects fall faster in vacuum"
            ],
            distractor_templates=[
                "Objects need continuous force to move at constant speed",
                "Acceleration and velocity are the same thing",
                "Mass affects falling speed in vacuum"
            ]
        ),
        "energy_conservation": MisconceptionPattern(
            name="Energy Conservation Misunderstanding",
            description="Incorrect understanding of energy transformation and conservation",
            examples=[
                "Energy is used up rather than transformed",
                "Potential energy depends on speed",
                "Energy can be created or destroyed"
            ],
            distractor_templates=[
                "Energy gets used up when work is done",
                "Higher speed means higher potential energy",
                "Energy can be created from nothing"
            ]
        )
    },
    "chemistry": {
        "atomic_structure": MisconceptionPattern(
            name="Atomic Structure Confusion",
            description="Misunderstanding of atomic composition and behavior",
            examples=[
                "Electrons orbit nucleus like planets",
                "All atoms of same element are identical",
                "Atoms can be seen with regular microscopes"
            ],
            distractor_templates=[
                "Electrons follow fixed orbital paths",
                "Isotopes don't exist for most elements",
                "Atoms are visible with optical microscopes"
            ]
        ),
        "chemical_bonding": MisconceptionPattern(
            name="Chemical Bonding Misconceptions",
            description="Incorrect ideas about how atoms bond",
            examples=[
                "All bonds are either 100% ionic or 100% covalent",
                "Electronegativity doesn't affect bond type",
                "Bond strength is unrelated to bond length"
            ],
            distractor_templates=[
                "Bonds are purely ionic or purely covalent",
                "All atoms bond with equal strength",
                "Longer bonds are always stronger"
            ]
        )
    },
    "biology": {
        "evolution": MisconceptionPattern(
            name="Evolution Misconceptions",
            description="Common misunderstandings about evolutionary processes",
            examples=[
                "Evolution is 'just a theory' (meaning guess)",
                "Organisms evolve on purpose/with intention",
                "Humans evolved from modern apes"
            ],
            distractor_templates=[
                "Evolution is unproven speculation",
                "Animals evolve to meet their needs",
                "Humans descended from chimpanzees"
            ]
        ),
        "genetics": MisconceptionPattern(
            name="Genetic Inheritance Confusion",
            description="Misunderstanding of how traits are inherited",
            examples=[
                "Dominant traits are more common",
                "Parents pass exact copies of traits",
                "One gene controls one trait always"
            ],
            distractor_templates=[
                "Dominant alleles are found in most populations",
                "Children are exact blends of parents",
                "Each trait is controlled by a single gene"
            ]
        )
    },
    "mathematics": {
        "algebra": MisconceptionPattern(
            name="Algebraic Reasoning Errors",
            description="Common mistakes in algebraic manipulation",
            examples=[
                "Variables represent specific unknown numbers",
                "You can't add variables with different letters",
                "Equations must always have number answers"
            ],
            distractor_templates=[
                "x always represents the same number",
                "You cannot add 2x + 3y",
                "Equations with variables have no solutions"
            ]
        ),
        "fractions": MisconceptionPattern(
            name="Fraction Misconceptions",
            description="Misunderstandings about fraction operations and meaning",
            examples=[
                "Larger denominators mean larger fractions",
                "When multiplying fractions, the answer is always bigger",
                "You can only add fractions with same denominators"
            ],
            distractor_templates=[
                "1/8 is larger than 1/4 because 8 > 4",
                "1/2 × 1/3 = 2/6 = 1/3, which is larger than 1/2",
                "1/3 + 1/4 cannot be calculated without converting"
            ]
        )
    }
}

def get_misconception_patterns(subject: str = None) -> Dict[str, MisconceptionPattern]:
    """
    Get misconception patterns for a specific subject or all subjects
    
    Args:
        subject: Subject name (physics, chemistry, biology, mathematics)
        
    Returns:
        Dictionary of misconception patterns
    """
    if subject and subject.lower() in MISCONCEPTION_PATTERNS:
        return MISCONCEPTION_PATTERNS[subject.lower()]
    return MISCONCEPTION_PATTERNS

def get_distractor_templates(topic: str, subject: str = None) -> List[str]:
    """
    Get distractor templates for a specific topic and subject
    
    Args:
        topic: Specific topic within subject
        subject: Subject name
        
    Returns:
        List of distractor templates
    """
    patterns = get_misconception_patterns(subject)
    
    if subject and subject.lower() in patterns:
        subject_patterns = patterns[subject.lower()]
        if topic in subject_patterns:
            return subject_patterns[topic].distractor_templates
    
    # Return general templates if specific ones not found
    all_templates = []
    for subject_data in patterns.values():
        for pattern in subject_data.values():
            all_templates.extend(pattern.distractor_templates)
    
    return all_templates[:5]  # Return first 5 as fallback
