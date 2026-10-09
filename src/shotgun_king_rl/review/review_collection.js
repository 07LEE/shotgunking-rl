const ammoInputs = { loaded: 'ammo_loaded', reserve: 'ammo_reserve' };
const combatFields = ['attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent', 'knockback_percent', 'grenades_remaining', 'melee_damage'];
const reviewInputs = { screen_kind: 'screenKindReviewed', ammo: 'ammoReviewed', combat_stats: 'combatReviewed' };
const token = document.querySelector('meta[name=review-token]').content, classes = ['empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king', 'special_knight'];
const $ = id => document.getElementById(id); let items = [], data, revision, selected = [0, 0], dirty = false, currentSession = '', currentReviewTab = 'pieces';
function remember() { try { localStorage.setItem('piece-review-selection', JSON.stringify({ session: currentSession, id: currentId })) } catch (e) { } }
function recalled() { try { return JSON.parse(localStorage.getItem('piece-review-selection')) || {} } catch (e) { return {} } }
function sessionItems() { return items.filter(item => item.session === currentSession) }
function pageReviewCount(item) { return Object.keys(reviewInputs).filter(section => item.reviews?.[section] === true).length }
function pageComplete(item) { return pageReviewCount(item) === Object.keys(reviewInputs).length }
function needsReview(item) { return currentReviewTab === 'pieces' ? !item.confirmed : !pageComplete(item) }
function firstPending(list) { return list.find(needsReview) || list[0] }
function otherSessionPending() {
    const sessions = [...new Set(items.map(item => item.session))];
    const index = sessions.indexOf(currentSession);
    const ordered = [...sessions.slice(index + 1), ...sessions.slice(0, index)];
    for (const session of ordered) {
        const pending = items.find(item => item.session === session && needsReview(item));
        if (pending) return pending;
    }
    return null;
}
function nextPending() {
    const list = sessionItems();
    const index = list.findIndex(item => item.id === currentId);
    return list.slice(index + 1).find(needsReview) || otherSessionPending();
}
async function moveTo(item) {
    if (item.session !== currentSession) {
        currentSession = item.session;
        $('sessions').value = currentSession;
    }
    currentId = item.id;
    remember();
    renderOptions();
    await load();
}
function reviewMarker(item) { if (currentReviewTab === 'pieces') return item.confirmed ? '✓' : '○'; const count = pageReviewCount(item); return count === Object.keys(reviewInputs).length ? '✓' : count > 0 ? '◐' : '○' }
function showReviewStatus() { if (!data) return; const item = items.find(candidate => candidate.id === currentId); status(currentReviewTab === 'pieces' ? (data.confirmed ? 'Piece labels approved.' : 'Piece labels are still a draft.') : (pageComplete(item) ? 'Page information reviewed.' : 'Page information review is incomplete.')) }
for (const c of classes) { const o = document.createElement('option'); o.value = c; o.textContent = c; $('label').append(o) }
async function api(url, options) { const r = await fetch(url, options); const a = await r.json(); if (!r.ok) throw Error(a.error || r.statusText); return a }
function status(s, error = false) {
    $('status').textContent = s;
    $('status').classList.toggle('error', error);
    if (error) $('status').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
function exclude(r, c) { return data.exclude_cells.some(x => x[0] === r && x[1] === c) }
function draw() {
    const grid = $('grid');
    grid.replaceChildren();
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const button = document.createElement('button');
            button.className = 'cell' + (exclude(r, c) ? ' excluded' : '') + (selected[0] === r && selected[1] === c ? ' selected' : '');
            const square = String.fromCharCode(97 + c) + (8 - r);
            button.title = square + ': ' + data.board[r][c];
            button.setAttribute('aria-label', button.title);
            const text = document.createElement('span');
            text.className = data.board[r][c] === 'empty' ? 'empty-label' : 'piece-label';
            text.textContent = data.board[r][c].replace('player_king', 'Player king').replace('white_king', 'White king').replace('special_knight', 'Special N');
            button.append(text);
            button.onclick = () => { selected = [r, c]; draw(); };
            grid.append(button);
        }
    }
    const [r, c] = selected;
    $('label').value = data.board[r][c];
    $('excluded').checked = exclude(r, c);
    $('locked').checked = data.screen_state.locked_cells.some(x => x[0] === r && x[1] === c);
    $('cellInfo').textContent = String.fromCharCode(97 + c) + (8 - r) + ': ' + data.board[r][c];
}
function labelView() {
    $('grid').classList.toggle('show-labels', $('showLabels').checked);
}
$('showLabels').onchange = labelView;
labelView();
function cropValue() { const parts = $('crop').value.split(',').map(x => x.trim()); if (parts.length !== 4 || parts.some(x => !/^\d+$/.test(x))) throw Error('Enter four nonnegative integer crop coordinates'); return parts.map(Number) }
function preview() { const crop = cropValue(); $('image').src = '/api/board?id=' + encodeURIComponent($('items').value) + '&crop=' + encodeURIComponent(JSON.stringify(crop)); if (data) draw(); }
async function load() { try { const a = await api('/api/item?id=' + encodeURIComponent($('items').value)); data = a.data; revision = a.revision; data.exclude_cells ??= []; data.screen_state ??= {}; data.screen_state.locked_cells ??= []; data.screen_state.reviews ??= {}; $('screenKind').value = data.screen_state.screen_kind ?? ''; for (const [key, id] of Object.entries(ammoInputs)) $(id).value = data.screen_state.ammo?.[key] ?? ''; for (const key of combatFields) $(key).value = data.screen_state.combat_stats?.[key] ?? ''; for (const [section, id] of Object.entries(reviewInputs)) $(id).checked = data.screen_state.reviews[section] === true; $('fullImage').src = '/api/screen?id=' + encodeURIComponent($('items').value); $('fullLink').href = $('fullImage').src; selected = [0, 0]; $('floor').value = data.floor ?? ''; $('crop').value = data.game_crop.join(', '); $('split').textContent = 'Session split: ' + data.split; preview(); draw(); dirty = false; showReviewStatus() } catch (e) { status(e.message, true) } }
$('image').onerror = () => status('Board preview failed. Check crop coordinates.');
$('items').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('items').value = currentId; return } currentId = $('items').value; remember(); renderOptions(); load() }; let currentId = '';
$('sessions').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('sessions').value = currentSession; return } currentSession = $('sessions').value; const list = sessionItems(); currentId = firstPending(list).id; remember(); renderOptions(); load() };
$('locked').onchange = () => { const [r, c] = selected; data.screen_state.locked_cells = data.screen_state.locked_cells.filter(x => x[0] !== r || x[1] !== c); if ($('locked').checked) data.screen_state.locked_cells.push([r, c]); dirty = true; draw() };
for (const id of [...Object.values(ammoInputs), ...combatFields, 'screenKind', ...Object.values(reviewInputs)]) $(id).oninput = () => { dirty = true };
$('label').onchange = () => { const [r, c] = selected; data.board[r][c] = $('label').value; dirty = true; draw() };
$('excluded').onchange = () => { const [r, c] = selected; data.exclude_cells = data.exclude_cells.filter(x => x[0] !== r || x[1] !== c); if ($('excluded').checked) data.exclude_cells.push([r, c]); dirty = true; draw() };
for (const id of ['floor', 'crop']) $(id).oninput = () => { dirty = true };
$('preview').onclick = () => { try { preview() } catch (e) { status(e.message) } };
async function save(confirmed) {
    const buttons = [$('draft'), $('approve')];
    buttons.forEach(button => { button.disabled = true; });
    try {
        const raw = $('floor').value;
        const floor = raw === '' ? null : Number(raw);
        if ((currentReviewTab === 'pieces' && confirmed && floor === null) || (floor !== null && (!Number.isInteger(floor) || floor < 1))) {
            switchReviewTab('screen');
            $('floor').setCustomValidity('Enter a positive floor number before approval.');
            $('floor').reportValidity();
            $('floor').focus();
            throw Error('Enter the floor number, then approve again.');
        }
        $('floor').setCustomValidity('');
        const ammo = Object.fromEntries(Object.entries(ammoInputs).map(([key, id]) => [key, $(id).value === '' ? null : Number($(id).value)]));
        const combat_stats = Object.fromEntries(combatFields.map(key => [key, $(key).value === '' ? null : Number($(key).value)]));
        if ([...Object.values(ammo), ...Object.values(combat_stats)].some(value => value !== null && (!Number.isInteger(value) || value < 0))) throw Error('Numeric values must be nonnegative integers.');
        const reviews = { ...data.screen_state.reviews, ...Object.fromEntries(Object.entries(reviewInputs).map(([section, id]) => [section, $(id).checked])) };
        const screen_kind = $('screenKind').value || null;
        if (reviews.screen_kind && screen_kind === null) throw Error('Select a screen kind before marking it reviewed.');
        const cards = data.screen_state.cards ?? { left: null, right: null };
        data.screen_state = { screen_kind, ammo, combat_stats, cards, locked_cells: data.screen_state.locked_cells, reviews };
        const boardConfirmed = currentReviewTab === 'pieces' ? confirmed : data.confirmed;
        status('Saving review…');
        const result = await api('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Review-Token': token },
            body: JSON.stringify({ id: currentId, revision, screen_state: data.screen_state, board: data.board, exclude_cells: data.exclude_cells, game_crop: cropValue(), floor, split: data.split, confirmed: boardConfirmed })
        });
        revision = result.revision;
        data.confirmed = boardConfirmed;
        dirty = false;
        const item = items.find(candidate => candidate.id === currentId);
        item.confirmed = boardConfirmed;
        item.reviews = reviews;
        renderOptions();
        if (!confirmed) {
            status(currentReviewTab === 'pieces' ? 'Piece draft saved; excluded from approved datasets.' : 'Page information saved; piece approval is unchanged.');
            return;
        }
        const next = nextPending();
        if (next) {
            await moveTo(next);
        } else {
            const remaining = items.filter(needsReview).length;
            if (remaining) status('No later unreviewed screen — ' + remaining + ' earlier screen' + (remaining === 1 ? '' : 's') + ' still need review.');
            else status(currentReviewTab === 'pieces' ? 'Piece review complete — all ' + items.length + ' screens approved.' : 'Page-information review complete — all ' + items.length + ' screens reviewed.');
        }
    } catch (error) {
        status('Not saved — ' + error.message, true);
    } finally {
        buttons.forEach(button => { button.disabled = false; });
    }
}
$('floor').addEventListener('input', () => $('floor').setCustomValidity(''));
$('draft').onclick = () => save(false); $('approve').onclick = () => save(true);
function renderOptions() { const list = sessionItems(); $('items').replaceChildren(); for (const item of list) { const o = document.createElement('option'); o.value = item.id; o.textContent = reviewMarker(item) + ' ' + item.id.split('/').pop(); $('items').append(o) } $('items').value = currentId; const index = list.findIndex(x => x.id === currentId); $('previous').disabled = index <= 0; $('next').disabled = index >= list.length - 1 && !otherSessionPending(); const count = section => list.filter(item => item.reviews?.[section]).length; $('progress').textContent = currentReviewTab === 'pieces' ? 'pieces ' + list.filter(x => x.confirmed).length + '/' + list.length + ' · screenshot ' + (index + 1) + '/' + list.length : 'page info ' + list.filter(pageComplete).length + '/' + list.length + ' · kind ' + count('screen_kind') + ' · ammo ' + count('ammo') + ' · stats ' + count('combat_stats') + ' · screenshot ' + (index + 1) + '/' + list.length }
async function navigate(delta) { const list = sessionItems(); const index = list.findIndex(x => x.id === currentId); const target = delta < 0 ? list[index - 1] : list[index + 1] || otherSessionPending(); if (!target) return; if (dirty && !confirm('Discard unsaved edits?')) return; await moveTo(target) }
$('previous').onclick = () => navigate(-1); $('next').onclick = () => navigate(1);
window.onbeforeunload = e => { if (dirty) { e.preventDefault(); e.returnValue = '' } };
(async () => { try { items = await api('/api/items'); if (!items.length) { status('No annotations. Run with --session to prepare inbox images.'); return } const sessions = [...new Set(items.map(x => x.session))]; const saved = recalled(); currentSession = sessions.includes(saved.session) ? saved.session : sessions[0]; for (const session of sessions) { const o = document.createElement('option'); o.value = session; o.textContent = session; $('sessions').append(o) } $('sessions').value = currentSession; const list = sessionItems(); currentId = list.some(x => x.id === saved.id) ? saved.id : firstPending(list).id; remember(); renderOptions(); await load() } catch (e) { status(e.message) } })();

$('predictState').onclick = async () => {
    $('predictState').disabled = true;
    try {
        const result = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Token': token }, body: JSON.stringify({ id: currentId, revision, game_crop: cropValue() }) });
        let count = 0;
        for (const [key, id] of Object.entries(ammoInputs)) if ($(id).value === '' && result.screen_state.ammo[key] !== null) { $(id).value = result.screen_state.ammo[key]; count++; $('ammoReviewed').checked = false; }
        for (const key of combatFields) if ($(key).value === '' && result.screen_state.combat_stats[key] !== null) { $(key).value = result.screen_state.combat_stats[key]; count++; $('combatReviewed').checked = false; }
        if (count) dirty = true;
        status(count ? 'Filled ' + count + ' fields. Review predictions before saving.' : 'No confident matches. Add reviewed examples with known values first.');
    } catch (error) { status('Prediction failed — ' + error.message, true); }
    finally { $('predictState').disabled = false; }
};

function switchReviewTab(name, focus = false) {
    currentReviewTab = name;
    const main = document.querySelector('main');
    main.classList.toggle('page-info-mode', name === 'screen');
    for (const [tab, panel, key] of [['piecesTab', 'piecesView', 'pieces'], ['screenTab', 'screenView', 'screen']]) {
        const active = name === key;
        $(tab).setAttribute('aria-selected', String(active));
        $(tab).tabIndex = active ? 0 : -1;
        $(panel).hidden = !active;
        $(key === 'pieces' ? 'piecesControls' : 'screenControls').hidden = !active;
        if (active && focus) $(tab).focus();
    }
    $('draft').textContent = name === 'pieces' ? 'Save draft' : 'Save page info';
    $('approve').textContent = name === 'pieces' ? 'Approve pieces ✓' : 'Save & next →';
    $('saveHelp').textContent = name === 'pieces' ? 'Check every piece label before approval.' : 'Each page-information section has its own review state. Saving this tab preserves piece approval.';
    if (items.length) { renderOptions(); showReviewStatus(); }
}
for (const [id, name] of [['piecesTab', 'pieces'], ['screenTab', 'screen']]) {
    $(id).onclick = () => {
        switchReviewTab(name);
        history.replaceState(null, '', name === 'screen' ? '#page-info' : '#pieces');
    };
    $(id).onkeydown = event => {
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            switchReviewTab(event.key === 'Home' ? 'pieces' : event.key === 'End' ? 'screen' : name === 'pieces' ? 'screen' : 'pieces', true);
        }
    };
}
switchReviewTab(location.hash === '#page-info' ? 'screen' : 'pieces');
