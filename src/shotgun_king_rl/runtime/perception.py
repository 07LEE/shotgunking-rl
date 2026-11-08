"""Construct the live readers from reviewed training references."""
from ..collection.combat_evaluation import build_combat_reader
from ..perception import OverlayScreenReader, PerceptionPipeline, RedPixelAmmoReader, TemplatePieceReader
from ..perception.combat import combat_state_contract
from ..perception.board_context import BoardContextReader
from ..perception.optional_zero import connect_optional_zero_readers
from ..perception.grenade_damage import RetainedGrenadeDamageReader
from ..perception.visibility_proposals import combat_field_readers, combat_visibility_titles


def build_live_pipeline(collection, melee_title_annotation=None, context_reader=None):
    combat, metadata = build_combat_reader(collection, melee_title_annotation)
    metadata['combat_state_contract'] = combat_state_contract()
    titles = combat_visibility_titles(combat_field_readers(combat))
    connect_optional_zero_readers(combat, titles, metadata)
    combat.grenade_damage_reader = RetainedGrenadeDamageReader(
        combat.grenade_damage_reader, visibility_titles=titles)
    metadata['readers']['grenade_damage_reader']['memory_policy'] = 'last_observed_in_current_run'
    metadata['readers']['grenade_damage_reader']['reuse_condition'] = 'verified_not_displayed'
    if context_reader is None:
        context_reader = BoardContextReader.from_collection(collection)
    metadata['readers']['board_context_reader'] = (
        context_reader.metadata() if isinstance(context_reader, BoardContextReader)
        else {'source': 'injected', 'reader': type(context_reader).__name__})
    return PerceptionPipeline(
        OverlayScreenReader(), TemplatePieceReader.from_collection(collection),
        RedPixelAmmoReader(), combat, context_reader,
    ), metadata
