"""Construct the live readers from reviewed training references."""
from ..collection.combat_evaluation import build_combat_reader
from ..perception import OverlayScreenReader, PerceptionPipeline, RedPixelAmmoReader, TemplatePieceReader
from ..perception.grenade_damage import RetainedGrenadeDamageReader


def build_live_pipeline(collection, melee_title_annotation=None, context_reader=None):
    combat, metadata = build_combat_reader(collection, melee_title_annotation)
    readers = {
        'attack': combat.attack_reader, 'range': combat.range_reader,
        'spread_degrees': combat.spread_reader, 'penetration_percent': combat.penetration_reader,
        'knockback_percent': combat.knockback_reader, 'melee_damage': combat.melee_reader,
        'grenade_damage': combat.grenade_damage_reader,
    }
    titles = [(field, row[:30]) for field, reader in readers.items()
              for value, row in reader.examples if field == 'range' or value > 0]
    combat.grenade_damage_reader = RetainedGrenadeDamageReader(
        combat.grenade_damage_reader, visibility_titles=titles)
    metadata['readers']['grenade_damage_reader']['memory_policy'] = 'last_observed_in_current_run'
    metadata['readers']['grenade_damage_reader']['reuse_condition'] = 'verified_not_displayed'
    return PerceptionPipeline(
        OverlayScreenReader(), TemplatePieceReader.from_collection(collection),
        RedPixelAmmoReader(), combat, context_reader,
    ), metadata
