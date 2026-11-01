"""Construct the live readers from reviewed training references."""
from ..collection.combat_evaluation import build_combat_reader
from ..perception import OverlayScreenReader, PerceptionPipeline, RedPixelAmmoReader, TemplatePieceReader
from ..perception.grenade_damage import RetainedGrenadeDamageReader


def build_live_pipeline(collection, melee_title_annotation=None, context_reader=None):
    combat, metadata = build_combat_reader(collection, melee_title_annotation)
    combat.grenade_damage_reader = RetainedGrenadeDamageReader(combat.grenade_damage_reader)
    metadata['readers']['grenade_damage_reader']['memory_policy'] = 'last_observed_in_current_run'
    return PerceptionPipeline(
        OverlayScreenReader(), TemplatePieceReader.from_collection(collection),
        RedPixelAmmoReader(), combat, context_reader,
    ), metadata
