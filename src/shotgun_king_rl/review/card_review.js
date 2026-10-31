const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name=review-token]').content;
let catalog, items, item, revision, currentId, dirty = false;
const slots = new Map();
async function api(path, payload) {
    const response = await fetch(path, payload ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Token': token }, body: JSON.stringify(payload) } : {});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || 'Request failed');
    return result;
}
function side(slot) { const [group, index] = slot.split(':'); return group === 'choice' ? Number(index) % 2 === 0 ? 'black' : 'white' : group === 'left' ? 'black' : 'white'; }
function box(slot) {
    const [group, raw] = slot.split(':'), index = Number(raw);
    if (group === 'choice') return [index % 2 === 0 ? 540 : 651, index < 2 ? 216 : 416];
    return [(group === 'left' ? 30 : 1062) + index % 2 * 96, 50 + Math.floor(index / 2) * 128];
}
function addSlot(slot, value, container) {
    const panel = document.createElement('section'); panel.className = 'slot';
    const title = document.createElement('strong'); title.textContent = slot + ' · ' + side(slot);
    const canvas = document.createElement('canvas'); canvas.width = 84; canvas.height = 116;
    const [x, y] = box(slot); canvas.getContext('2d').drawImage($('screen'), x, y, 84, 116, 0, 0, 84, 116);
    const select = document.createElement('select'); select.setAttribute('aria-label', slot + ' card name');
    const options = [{ key: '', text: 'Unknown — needs review' }];
    if (!slot.startsWith('choice:')) options.push({ key: 'empty', text: 'Empty — no card' });
    for (const entry of catalog.entries.filter(entry => entry.side === side(slot))) options.push({ key: entry.key, text: (entry.names.ko || entry.names.en) + ' / ' + entry.names.en });
    for (const value of options) { const option = document.createElement('option'); option.value = value.key; option.textContent = value.text; select.append(option); }
    select.value = value === 'unknown' || value == null ? '' : value;
    const effects = document.createElement('textarea'); effects.placeholder = 'Optional: effect description verified from the game'; effects.setAttribute('aria-label', slot + ' effect description');
    const effectState = document.createElement('span'); effectState.className = 'help';
    const showEffects = () => { const entry = catalog.entries.find(entry => entry.key === select.value); effects.value = entry?.effects?.description || ''; effectState.textContent = entry?.effects ? 'Reviewed effect description' : 'Effects unknown'; };
    select.onchange = () => { showEffects(); dirty = true; }; effects.oninput = () => { dirty = true; }; showEffects();
    panel.append(title, canvas, select, effectState, effects); container.append(panel); slots.set(slot, { select, effects });
}
async function load() {
    catalog = await api('/api/card-catalog');
    const result = await api('/api/item?id=' + encodeURIComponent($('items').value)); item = result.data; revision = result.revision;
    $('screen').src = '/api/screen?id=' + encodeURIComponent($('items').value) + '&revision=' + encodeURIComponent(revision);
    await $('screen').decode(); slots.clear(); $('choices').replaceChildren(); $('persistent').replaceChildren();
    $('kindReviewed').checked = item.screen_state.reviews.screen_kind;
    $('split').textContent = 'Session: ' + item.session_id + ' · Split: ' + item.split + (item.split === 'train' ? ' · Confirmed labels can supply references.' : ' · Held-out labels never supply recognition references.');
    const choices = item.screen_state.card_choices;
    $('choiceHeading').hidden = item.screen_state.screen_kind !== 'card_selection';
    if (item.screen_state.screen_kind === 'card_selection') for (let i = 0; i < 4; i++) addSlot('choice:' + i, choices?.[i], $('choices'));
    for (const group of ['left', 'right']) for (let i = 0; i < 10; i++) {
        const values = item.screen_state.cards[group];
        addSlot(group + ':' + i, values?.[i] ?? (item.screen_state.reviews.cards && values ? 'empty' : null), $('persistent'));
    }
    currentId = $('items').value;
    $('status').textContent = item.screen_state.reviews.cards ? 'Card labels reviewed.' : 'Card labels need review.'; dirty = false;
}
async function save(confirmed) {
    const cards = { left: [], right: [] }, choices = item.screen_state.screen_kind === 'card_selection' ? [] : null, descriptions = {};
    for (const [slot, controls] of slots) {
        const [group] = slot.split(':'), value = controls.select.value || null;
        if (group === 'choice') choices.push(value); else cards[group].push(value);
        if (value && value !== 'empty' && controls.effects.value.trim()) {
            const text = controls.effects.value.trim();
            if (descriptions[value] && descriptions[value] !== text) throw Error('Use one effect description for repeated copies of a card.');
            descriptions[value] = text;
        }
    }
    await api('/api/card-review', { id: $('items').value, revision, catalog_revision: catalog.revision, confirmed, screen_kind_reviewed: $('kindReviewed').checked, cards, card_choices: choices, effect_descriptions: descriptions });
    await load(); $('status').textContent = confirmed ? 'Card labels confirmed. Training references updated where eligible.' : 'Card draft saved.';
}
function run(action) { return async () => { try { await action(); } catch (error) { $('status').textContent = error.message; } }; }
$('draft').onclick = run(() => save(false)); $('confirm').onclick = run(() => save(true));
$('emptySlots').onclick = () => { for (const [slot, controls] of slots) if (!slot.startsWith('choice:') && !controls.select.value) { controls.select.value = 'empty'; controls.select.onchange(); } };
$('items').onchange = run(async () => { if (dirty && !confirm('Discard unsaved card edits?')) { $('items').value = currentId; return; } await load(); });
$('predict').onclick = run(async () => {
    const result = await api('/api/predict', { id: $('items').value, revision, game_crop: item.game_crop });
    const proposal = result.prediction.card_recognition;
    let count = 0;
    for (const match of proposal.matches) { const control = slots.get(match.slot); if (control && !control.select.value) { control.select.value = match.key; control.select.onchange(); count++; } }
    $('status').textContent = count ? 'Filled ' + count + ' card slots. Review before confirming.' : 'No confident reviewed references. Label cards to establish training references.';
});
$('next').onclick = run(async () => { if (dirty && !confirm('Discard unsaved card edits?')) return; $('items').selectedIndex = ($('items').selectedIndex + 1) % items.length; await load(); });
window.onbeforeunload = event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } };
run(async () => {
    items = (await api('/api/items')).filter(entry => ['gameplay', 'card_selection'].includes(entry.screen_kind));
    for (const entry of items) { const option = document.createElement('option'); option.value = entry.id; option.textContent = (entry.reviews.cards ? '✓ ' : '') + entry.session + ' / ' + entry.id.split('/').pop(); $('items').append(option); }
    if (!items.length) throw Error('No prepared gameplay or card-selection screenshots.');
    const requested = new URLSearchParams(location.search).get('id');
    if (items.some(entry => entry.id === requested)) $('items').value = requested;
    await load();
})();
