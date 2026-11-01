"""Use conservative optional-stat readers for collection proposals."""

from ..perception.knockback import KnockbackReader
from ..perception.penetration import PenetrationReader
from ..perception.melee import MeleeDamageReader
from ..perception.title_references import load_melee_title_reference


def melee_title_reference(collection, exclude=None):
    """Use the same selected and validated title as the runtime factory."""
    reference, _ = load_melee_title_reference(collection, exclude=exclude)
    return reference


def propose_optional_combat(screen, collection, exclude=None):
    """Keep absent rows unknown unless a reviewed zero-panel match proves zero."""
    readers = {
        'penetration_percent': PenetrationReader.from_collection(collection, exclude=exclude),
        'knockback_percent': KnockbackReader.from_collection(collection, exclude=exclude),
        'melee_damage': MeleeDamageReader.from_collection(collection, exclude=exclude, title_reference=melee_title_reference(collection, exclude)),
    }
    values, evidence = {}, {}
    for field, reader in readers.items():
        result = reader.read(screen)
        values[field] = result.value if result.is_accepted else None
        evidence[field] = {'source': result.source, 'reason': result.reason, 'confidence': result.confidence}
    return values, evidence
