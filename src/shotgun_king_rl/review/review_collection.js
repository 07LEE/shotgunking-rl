const ammoInputs = { loaded: 'ammo_loaded', reserve: 'ammo_reserve' };
const combatFields = ['attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent', 'knockback_percent', 'grenades_remaining', 'grenade_damage', 'melee_damage'];
const reviewInputs = { screen_kind: 'screenKindReviewed', ammo: 'ammoReviewed', combat_stats: 'combatReviewed' };
const independentReviewInputs = { soul: 'soulReviewed' };
const soulPieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king', 'special_knight'];
const token = document.querySelector('meta[name=review-token]').content, classes = ['ambiguous_royal', 'empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king', 'special_knight'];
const $ = id => document.getElementById(id); let editingSections = new Set(), loading = false, saving = false, loadedId = '', currentId = '';
const expandedSessions = new Set();
const reviewTasks = [
    { key: 'pieces', tab: 'piecesTab', label: 'Pieces' },
    { key: 'screen_kind', tab: 'screenTab', label: 'Screen type' },
    { key: 'ammo', tab: 'ammoTab', label: 'Ammunition' },
    { key: 'combat_stats', tab: 'combatTab', label: 'Combat stats' },
    { key: 'soul', tab: 'soulTab', label: 'Harvested souls' },
];
function currentTask() { return reviewTasks.find(task => task.key === currentReviewTab) }
function taskApplies(item) { return currentReviewTab === 'pieces' ? pieceApplicable(item) : currentReviewTab === 'screen_kind' || item.screen_kind === 'gameplay' }

let items = [], data, revision, selected = [0, 0], dirty = false, currentSession = '', currentReviewTab = 'pieces', pendingPieceApprovalId = null;
function remember() { try { localStorage.setItem('piece-review-selection', JSON.stringify({ session: currentSession, id: currentId })) } catch (e) { } }
function recalled() { try { return JSON.parse(localStorage.getItem('piece-review-selection')) || {} } catch (e) { return {} } }
function sessionItems() { return items.filter(item => item.session === currentSession) }
function nonGameplay(kind) { return Boolean(kind && kind !== 'gameplay') }
function pieceApplicable(item) { return !nonGameplay(item.screen_kind) }
function needsReview(item) { return taskApplies(item) && !(currentReviewTab === 'pieces' ? item.confirmed : item.reviews?.[currentReviewTab]) }
function relevantItems() { return sessionItems().filter(taskApplies) }
function visibleItems() {
    return relevantItems().filter(item => $('showCompleted').checked || needsReview(item) || (item.id === currentId && (dirty || editingSections.size || pendingPieceApprovalId === currentId)));
}
function firstPending(list) { return list.find(needsReview) || ($('showCompleted').checked ? list[0] : null) }
function savedItem() { return items.find(item => item.id === currentId) }
function sectionReviewed(section) { const item = savedItem(); return Boolean(item && (section === 'pieces' ? item.confirmed : item.reviews?.[section])) }
function sectionLocked(section) { return sectionReviewed(section) && !editingSections.has(section) }
function setDisabled(ids, disabled) { for (const id of ids) $(id).disabled = disabled }
function syncReviewLocks() {
    const item = savedItem();
    $('showCompleted').disabled = saving;
    for (const task of reviewTasks) $(task.tab).disabled = saving;
    for (const button of $('sessionList').querySelectorAll('button')) button.disabled = saving;
    const unavailable = !data || loadedId !== currentId || loading || saving || document.querySelector('main').hidden;
    const piecesLocked = unavailable || sectionLocked('pieces');
    setDisabled(['label', 'pieceSide', 'locked', 'excluded', 'moatActive'], piecesLocked);
    for (const cell of $('grid').children) cell.disabled = piecesLocked;
    if (!unavailable) $('pieceSide').disabled = piecesLocked || ['empty', 'player_king'].includes(data.board[selected[0]][selected[1]]);
    setDisabled(['screenKind', 'screenKindReviewed'], unavailable || sectionLocked('screen_kind'));
    setDisabled([...Object.values(ammoInputs), 'ammoReviewed'], unavailable || sectionLocked('ammo'));
    setDisabled([...combatFields, 'combatReviewed'], unavailable || sectionLocked('combat_stats'));
    const soulLocked = unavailable || sectionLocked('soul');
    setDisabled(['soulReviewed', 'addSoul'], soulLocked);
    for (const control of $('soulSlots').querySelectorAll('select, button')) control.disabled = soulLocked;
    $('floor').disabled = unavailable || (sectionLocked('screen_kind') && data.floor !== null);
    $('crop').disabled = unavailable || ['pieces', ...Object.keys(reviewInputs), 'soul'].some(sectionLocked);
    $('preview').disabled = unavailable || $('crop').disabled;
    $('predictState').disabled = unavailable || sectionLocked(currentReviewTab);
    $('editCrop').hidden = !['pieces', ...Object.keys(reviewInputs), 'soul'].some(sectionReviewed);
    $('editCrop').disabled = unavailable;
    const editable = pendingPieceApprovalId === currentId || !sectionLocked(currentReviewTab);
    setDisabled(['draft', 'approve'], unavailable || !item || !editable);
    $('reviewSectionStatus').replaceChildren();
    if (!item || !data || loadedId !== currentId || document.querySelector('main').hidden) return;
    const sections = [[currentReviewTab, currentTask().label]];
    for (const [key, label] of sections) {
        const complete = sectionLocked(key);
        const chip = document.createElement('span');
        chip.className = 'section-state' + (complete ? ' complete' : '');
        const text = document.createElement('span');
        text.textContent = (complete ? '✓ ' : '○ ') + label + (complete ? ' · reviewed' : editingSections.has(key) ? ' · editing, review again' : ' · needs review');
        chip.append(text);
        if (complete) {
            const edit = document.createElement('button'); edit.type = 'button'; edit.textContent = 'Edit'; edit.disabled = unavailable;
            edit.setAttribute('aria-label', 'Edit reviewed ' + label.toLowerCase());
            edit.onclick = () => reopenSection(key); chip.append(edit);
        }
        $('reviewSectionStatus').append(chip);
    }
}

function reopenSection(section, focus = true) {
    editingSections.add(section);
    if (section === 'pieces') data.confirmed = false;
    else $(({ ...reviewInputs, ...independentReviewInputs })[section]).checked = false;
    dirty = true;
    renderOptions();
    if (focus) $(({ pieces: 'label', screen_kind: 'screenKind', ammo: 'ammo_loaded', combat_stats: 'attack', soul: 'addSoul' })[section]).focus();
    status('Editing a reviewed section. Save and review it again when finished.');
}
async function selectVisible(reload = false) {
    const list = visibleItems();
    const target = list.find(item => item.id === currentId) || firstPending(list) || list[0] || items.find(needsReview);
    if (target) {
        if (target.id !== currentId || reload) await moveTo(target);
        else renderOptions();
    } else { currentId = ''; data = null; renderOptions(); }
}

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
    return [...list.slice(index + 1), ...list.slice(0, index)].find(needsReview) || otherSessionPending();
}
async function moveTo(item) {
    if (item.id !== currentId) pendingPieceApprovalId = null;
    if (item.session !== currentSession) {
        currentSession = item.session;
        expandedSessions.add(currentSession);
    }
    currentId = item.id;
    remember();
    renderOptions();
    await load();
}
function showReviewStatus() { if (data && !document.querySelector('main').hidden) status('') }
for (const c of classes) { const o = document.createElement('option'); o.value = c; o.textContent = c === 'ambiguous_royal' ? 'Queen / white king (unresolved)' : c; $('label').append(o) }
async function api(url, options) { const r = await fetch(url, options); const a = await r.json(); if (!r.ok) throw Error(a.error || r.statusText); return a }
function status(s, error = false) {
    $('status').textContent = s;
    $('status').classList.toggle('error', error);
    if (error) $('status').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
function exclude(r, c) { return data.exclude_cells.some(x => x[0] === r && x[1] === c) }
function draw() {
    if (!data) return;
    const grid = $('grid');
    grid.replaceChildren();
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const button = document.createElement('button');
            button.className = 'cell' + (exclude(r, c) ? ' excluded' : '') + (selected[0] === r && selected[1] === c ? ' selected' : '');
            const square = String.fromCharCode(97 + c) + (8 - r);
            button.classList.add(data.terrain[r][c] === 'moat' ? 'moat' : 'plain');
            button.title = square + ': ' + data.board[r][c] + ' · ' + (data.piece_sides[r][c] ?? 'no piece') + ' · ' + data.terrain[r][c];
            button.setAttribute('aria-label', button.title);
            const text = document.createElement('span');
            text.className = data.board[r][c] === 'empty' ? 'empty-label' : 'piece-label';
            text.textContent = (data.piece_sides[r][c] === 'ally' && data.board[r][c] !== 'player_king' ? 'Ally ' : '') + data.board[r][c].replace('ambiguous_royal', 'Queen / king?').replace('player_king', 'Player king').replace('white_king', 'White king').replace('special_knight', 'Special N');
            button.append(text);
            button.onclick = () => { selected = [r, c]; draw(); };
            grid.append(button);
        }
    }
    const [r, c] = selected;
    $('label').value = data.board[r][c];
    $('pieceSide').value = data.piece_sides[r][c] ?? '';
    $('pieceSide').disabled = ['empty', 'player_king'].includes(data.board[r][c]);
    $('excluded').checked = exclude(r, c);
    $('locked').checked = data.screen_state.locked_cells.some(x => x[0] === r && x[1] === c);
    $('cellInfo').textContent = String.fromCharCode(97 + c) + (8 - r) + ': ' + data.board[r][c] + ' · ' + (data.piece_sides[r][c] ?? 'no piece') + ' · ' + data.terrain[r][c];
    if (data.board[r][c] === 'ambiguous_royal') $('cellInfo').textContent += ' · Select queen or white king, or exclude this cell from training.';
    syncReviewLocks();
}
function labelView() {
    $('grid').classList.toggle('show-labels', $('showLabels').checked);
}
$('showLabels').onchange = labelView;
labelView();
function cropValue() { const parts = $('crop').value.split(',').map(x => x.trim()); if (parts.length !== 4 || parts.some(x => !/^\d+$/.test(x))) throw Error('Enter four nonnegative integer crop coordinates'); return parts.map(Number) }
function preview() { const crop = cropValue(); $('image').src = '/api/board?id=' + encodeURIComponent(currentId) + '&crop=' + encodeURIComponent(JSON.stringify(crop)); if (data) draw(); }
function renderSoulSlots(souls) {
    $('soulSlots').replaceChildren();
    for (const piece of souls || []) {
        const row = document.createElement('div');
        row.className = 'soul-slot';
        const select = document.createElement('select');
        select.setAttribute('aria-label', 'Harvested soul piece');
        const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose piece'; select.append(placeholder);
        for (const value of soulPieces) { const option = document.createElement('option'); option.value = value; option.textContent = value.replace('_', ' '); select.append(option); }
        select.value = piece;
        select.onchange = () => { $('soulReviewed').checked = true; dirty = true; };
        const remove = document.createElement('button');
        remove.type = 'button'; remove.textContent = '×'; remove.setAttribute('aria-label', 'Remove soul');
        remove.onclick = () => { row.remove(); $('soulReviewed').checked = true; dirty = true; };
        row.append(select, remove); $('soulSlots').append(row);
    }
}
function soulValues() { return [...$('soulSlots').querySelectorAll('select')].map(select => select.value) }
async function load() {
    const id = currentId;
    loading = true; syncReviewLocks();
    try {
        const a = await api('/api/item?id=' + encodeURIComponent(id));
        if (id !== currentId) return;
        data = a.data; loadedId = id; editingSections.clear();
        if (!Number.isInteger(data.moat_row)) throw Error('Restart the review server to use moat card controls.');
        revision = a.revision;
        $('moatActive').value = data.moat_active === true ? 'true' : data.moat_active === false ? 'false' : '';
        data.exclude_cells ??= []; data.screen_state ??= {}; data.screen_state.locked_cells ??= []; data.screen_state.reviews ??= {};
        const item = savedItem(); item.confirmed = data.confirmed; item.screen_kind = data.screen_state.screen_kind; item.reviews = { ...data.screen_state.reviews };
        $('screenKind').value = data.screen_state.screen_kind ?? '';
        for (const [key, input] of Object.entries(ammoInputs)) $(input).value = data.screen_state.ammo?.[key] ?? '';
        for (const key of combatFields) $(key).value = data.screen_state.combat_stats?.[key] ?? '';
        for (const [section, input] of Object.entries({ ...reviewInputs, ...independentReviewInputs })) $(input).checked = data.screen_state.reviews[section] === true;
        renderSoulSlots(data.screen_state.souls);
        $('fullImage').src = '/api/screen?id=' + encodeURIComponent(id); $('fullLink').href = $('fullImage').src; $('openOriginal').href = $('fullImage').src;
        selected = [0, 0]; $('floor').value = data.floor ?? ''; $('crop').value = data.game_crop.join(', ');
        $('split').textContent = 'Session split: ' + data.split;
        dirty = false; syncScreenKind(); preview(); draw();
        renderOptions(); showReviewStatus();
    } catch (e) { status(e.message, true); }
    finally { if (id === currentId) { loading = false; syncReviewLocks(); } }
}
$('image').onerror = () => status('Board preview failed. Check crop coordinates.');
async function selectReview(item) {
    if (saving || item.id === currentId) return;
    if (dirty && !confirm('Discard unsaved edits?')) return;
    pendingPieceApprovalId = null;
    await moveTo(item);
}
$('showCompleted').onchange = async () => {
    if (dirty && !confirm('Discard unsaved edits?')) { $('showCompleted').checked = !$('showCompleted').checked; return; }
    dirty = false; editingSections.clear(); pendingPieceApprovalId = null;
    await selectVisible(true); remember();
};
$('locked').onchange = () => { const [r, c] = selected; data.screen_state.locked_cells = data.screen_state.locked_cells.filter(x => x[0] !== r || x[1] !== c); if ($('locked').checked) data.screen_state.locked_cells.push([r, c]); dirty = true; draw() };
for (const id of [...Object.values(ammoInputs), ...combatFields, 'screenKind', ...Object.values(reviewInputs), ...Object.values(independentReviewInputs)]) $(id).oninput = () => { dirty = true };
$('addSoul').onclick = () => { const souls = soulValues(); souls.push(''); renderSoulSlots(souls); $('soulReviewed').checked = true; dirty = true; $('soulSlots').lastElementChild.querySelector('select').focus(); };
$('screenKind').addEventListener('change', syncScreenKind);
$('label').onchange = () => { const [r, c] = selected; data.board[r][c] = $('label').value; data.piece_sides[r][c] = data.board[r][c] === 'empty' ? null : data.board[r][c] === 'player_king' ? 'ally' : data.piece_sides[r][c] ?? 'enemy'; dirty = true; draw() };
$('pieceSide').onchange = () => { const [r, c] = selected; data.piece_sides[r][c] = $('pieceSide').value || 'unknown'; dirty = true; draw() };
$('moatActive').onchange = () => { const value = $('moatActive').value; data.moat_active = value === '' ? null : value === 'true'; data.terrain = Array.from({ length: 8 }, (_, row) => Array(8).fill(data.moat_active === null ? 'unknown' : data.moat_active && row === data.moat_row ? 'moat' : 'normal')); dirty = true; draw() };
$('excluded').onchange = () => { const [r, c] = selected; data.exclude_cells = data.exclude_cells.filter(x => x[0] !== r || x[1] !== c); if ($('excluded').checked) data.exclude_cells.push([r, c]); dirty = true; draw() };
for (const id of ['floor', 'crop']) $(id).oninput = () => { dirty = true };
$('preview').onclick = () => { try { preview() } catch (e) { status(e.message) } };
async function save(confirmed) {
    if (!data || saving || $('approve').disabled && $('draft').disabled) return;
    saving = true; syncReviewLocks();
    try {
        const raw = $('floor').value;
        const floor = raw === '' ? null : Number(raw);
        const invalidFloor = floor !== null && (!Number.isInteger(floor) || floor < 1);
        if (currentReviewTab === 'pieces' && confirmed && (floor === null || invalidFloor)) {
            pendingPieceApprovalId = currentId;
            switchReviewTab('screen');
            $('floor').setCustomValidity('Enter a positive floor number before approval.');
            $('floor').reportValidity();
            $('floor').focus();
            status('Enter the floor, then save page info to approve these pieces and continue.');
            return;
        }
        if (invalidFloor) {
            $('floor').setCustomValidity('Enter a positive floor number.');
            $('floor').reportValidity();
            $('floor').focus();
            throw Error('Enter a positive floor number.');
        }
        $('floor').setCustomValidity('');
        const ammo = Object.fromEntries(Object.entries(ammoInputs).map(([key, id]) => [key, $(id).value === '' ? null : Number($(id).value)]));
        const combat_stats = Object.fromEntries(combatFields.map(key => [key, $(key).value === '' ? null : Number($(key).value)]));
        if ([...Object.values(ammo), ...Object.values(combat_stats)].some(value => value !== null && (!Number.isInteger(value) || value < 0))) throw Error('Numeric values must be nonnegative integers.');
        const screen_kind = $('screenKind').value || null;
        const completingPieceApproval = currentReviewTab !== 'pieces' && pendingPieceApprovalId === currentId;
        const reviews = { ...data.screen_state.reviews, ...Object.fromEntries(Object.entries({ ...reviewInputs, ...independentReviewInputs }).map(([section, id]) => [section, $(id).checked])) };
        if (confirmed && currentReviewTab !== 'pieces' && !completingPieceApproval) {
            reviews[currentReviewTab] = true;
        }
        if (reviews.screen_kind && screen_kind === null) throw Error('Select a screen kind before marking it reviewed.');
        const cards = data.screen_state.cards ?? { left: null, right: null };
        const selectedSouls = soulValues();
        if (reviews.soul && selectedSouls.some(piece => !piece)) throw Error('Select a piece for every harvested soul slot.');
        const souls = reviews.soul ? selectedSouls : null;
        data.screen_state = { screen_kind, ammo, combat_stats, cards, card_choices: data.screen_state.card_choices ?? null, card_reviewed_slots: data.screen_state.card_reviewed_slots ?? [], souls, locked_cells: data.screen_state.locked_cells, reviews };
        const boardConfirmed = currentReviewTab === 'pieces' ? confirmed : completingPieceApproval ? true : data.confirmed;
        const pieceReviewed = (currentReviewTab === 'pieces' && confirmed) || completingPieceApproval;
        if (boardConfirmed && screen_kind === 'gameplay') {
            for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
                if (data.board[r][c] === 'ambiguous_royal' && !exclude(r, c)) {
                    selected = [r, c];
                    switchReviewTab('pieces');
                    draw();
                    $('label').focus();
                    throw Error('Select queen or white king for the highlighted cell, or exclude it from training.');
                }
            }
        }
        status('Saving review…');
        const result = await api('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Review-Token': token },
            body: JSON.stringify({ id: currentId, revision, screen_state: data.screen_state, board: data.board, piece_sides: data.piece_sides, terrain: data.terrain, moat_active: data.moat_active, exclude_cells: data.exclude_cells, game_crop: cropValue(), floor, split: data.split, confirmed: boardConfirmed, piece_reviewed: pieceReviewed })
        });
        revision = result.revision;
        data.confirmed = result.confirmed;
        data.piece_review = result.piece_review;
        dirty = false;
        const item = items.find(candidate => candidate.id === currentId);
        item.confirmed = result.confirmed;
        item.piece_review = result.piece_review;
        item.screen_kind = screen_kind;
        item.reviews = { ...reviews };
        for (const [section, input] of Object.entries({ ...reviewInputs, ...independentReviewInputs })) $(input).checked = reviews[section] === true;
        editingSections.clear();
        renderOptions();
        if (completingPieceApproval) {
            pendingPieceApprovalId = null;
            if (!result.confirmed) {
                status('Page information saved; this screen does not require piece review.');
                return;
            }
            switchReviewTab('pieces');
            const nextPiece = nextPending();
            if (nextPiece) await moveTo(nextPiece);
            else status('Piece review complete — all applicable screens approved.');
            return;
        }
        if (currentReviewTab === 'pieces') pendingPieceApprovalId = null;
        if (!confirmed) {
            const message = currentReviewTab === 'pieces'
                ? 'Piece draft saved; excluded from approved datasets.'
                : nonGameplay(screen_kind)
                    ? 'Page information saved; this screen is excluded from piece review.'
                    : 'Page information saved; piece approval is unchanged.';
            status(message);
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
        saving = false; renderOptions();
    }
}
$('floor').addEventListener('input', () => $('floor').setCustomValidity(''));
$('editCrop').onclick = () => {
    if (!confirm('Changing the crop requires reviewing this screenshot again. Continue?')) return;
    for (const key of ['pieces', 'screen_kind', 'ammo', 'combat_stats', 'soul']) reopenSection(key, false);
    $('crop').focus();
};
$('draft').onclick = () => save(false); $('approve').onclick = () => save(true);
function renderQueue() {
    const showCompleted = $('showCompleted').checked;
    const relevant = items.filter(taskApplies);
    const pending = relevant.filter(needsReview).length;
    $('queueTotals').textContent = pending + ' pending · ' + (relevant.length - pending) + ' reviewed';
    $('sessionList').replaceChildren();
    for (const session of [...new Set(relevant.map(item => item.session))]) {
        const all = relevant.filter(item => item.session === session);
        const list = all.filter(item => showCompleted || needsReview(item) || (item.id === currentId && (dirty || editingSections.size || pendingPieceApprovalId === currentId)));
        if (!list.length) continue;
        const group = document.createElement('details'); group.className = 'queue-session'; group.dataset.session = session;
        group.open = session === currentSession || expandedSessions.has(session);
        group.ontoggle = () => { if (group.isConnected) { if (group.open) expandedSessions.add(session); else expandedSessions.delete(session); } };
        const summary = document.createElement('summary');
        const name = document.createElement('span'); name.textContent = session;
        const count = document.createElement('span'); count.className = 'queue-count'; count.textContent = all.filter(needsReview).length + ' pending';
        summary.append(name, count); group.append(summary);
        const entries = document.createElement('ul');
        for (const item of list) {
            const entry = document.createElement('li'), button = document.createElement('button');
            const active = item.id === currentId;
            button.type = 'button'; button.className = 'queue-item' + (active ? ' active' : '');
            button.dataset.id = item.id; button.disabled = saving;
            if (active) button.setAttribute('aria-current', 'true');
            const title = document.createElement('span'); title.className = 'queue-item-name';
            const filename = item.id.split('/').pop().replace(/\.json$/, '');
            title.textContent = filename.replace(/^Screenshot From /, ''); button.title = filename;
            const state = document.createElement('span'); state.className = 'queue-item-state';
            state.textContent = (active && editingSections.size ? 'Editing · review again' : needsReview(item) ? '○ Needs review' : '✓ Reviewed · read-only') + (item.floor ? ' · Floor ' + item.floor : '');
            button.append(title, state); button.onclick = () => selectReview(item);
            entry.append(button); entries.append(entry);
        }
        group.append(entries); $('sessionList').append(group);
    }
    const selected = $('sessionList').querySelector('[aria-current="true"]');
    if (selected) {
        const bounds = $('sessionList').getBoundingClientRect(), row = selected.getBoundingClientRect();
        if (row.bottom > bounds.bottom) $('sessionList').scrollTop += row.bottom - bounds.bottom + 8;
        else if (row.top < bounds.top) $('sessionList').scrollTop -= bounds.top - row.top + 8;
    }
    if (!$('sessionList').children.length) {
        const empty = document.createElement('p'); empty.className = 'queue-empty';
        empty.textContent = showCompleted ? 'No screenshots for this review category.' : 'No pending reviews. Enable “Show completed” to inspect previous work.';
        $('sessionList').append(empty);
    }
}
function renderOptions() {
    const all = relevantItems(), list = visibleItems();
    const index = list.findIndex(item => item.id === currentId), remaining = all.filter(needsReview).length;
    $('previous').disabled = index <= 0 || saving;
    $('next').disabled = saving || (index >= list.length - 1 && !otherSessionPending());
    renderQueue();
    const current = list.find(item => item.id === currentId);
    document.querySelector('main').hidden = !current;
    $('reviewSummaryText').textContent = !list.length
        ? 'No pending ' + currentTask().label.toLowerCase() + ' reviews. Select another image or show completed work.'
        : remaining + ' pending · ' + (all.length - remaining) + ' reviewed' + (editingSections.size ? ' · Editing reviewed work; review it again before saving.' : current && !needsReview(current) ? ' · Read-only. Choose Edit to make a correction.' : '');
    $('currentScreenshot').textContent = current ? current.session + ' / ' + current.id.split('/').pop().replace(/\.json$/, '') : '';
    $('progress').textContent = (all.length - remaining) + '/' + all.length + ' reviewed · screenshot ' + Math.max(0, index + 1) + '/' + list.length;
    syncReviewLocks();
}
async function navigate(delta) {
    if (saving) return;
    const list = visibleItems(), index = list.findIndex(item => item.id === currentId);
    const target = delta < 0 ? list[index - 1] : list[index + 1] || otherSessionPending();
    if (!target) return;
    if (dirty && !confirm('Discard unsaved edits?')) return;
    await moveTo(target);
}
$('previous').onclick = () => navigate(-1); $('next').onclick = () => navigate(1);
window.onbeforeunload = e => { if (dirty) { e.preventDefault(); e.returnValue = '' } };
(async () => {
    try {
        items = await api('/api/items');
        if (!items.length) { renderOptions(); status('No prepared images. Process the inbox first.'); return; }
        const sessions = [...new Set(items.map(item => item.session))], saved = recalled();
        const requestedSession = new URLSearchParams(location.search).get('session');
        currentSession = sessions.includes(requestedSession) ? requestedSession : sessions.includes(saved.session) && items.some(item => item.session === saved.session && needsReview(item)) ? saved.session : items.find(needsReview)?.session || sessions[0];
        expandedSessions.add(currentSession);
        const list = visibleItems(); currentId = list.some(item => item.id === saved.id) ? saved.id : firstPending(list)?.id || '';
        if (requestedSession) history.replaceState(null, '', location.pathname + location.hash);
        remember(); renderOptions(); if (currentId) await load();
    } catch (e) { status(e.message, true); }
})();

$('predictState').onclick = async () => {
    $('predictState').disabled = true;
    try {
        const result = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Token': token }, body: JSON.stringify({ id: currentId, revision, game_crop: cropValue() }) });
        let count = 0;
        if (currentReviewTab === 'screen_kind' && !sectionLocked('screen_kind') && $('screenKind').value === '' && result.screen_state.screen_kind !== null) { $('screenKind').value = result.screen_state.screen_kind; $('screenKindReviewed').checked = false; syncScreenKind(); count++; }
        if (currentReviewTab === 'screen_kind' && !$('floor').disabled && $('floor').value === '' && result.floor !== null) { $('floor').value = result.floor; count++; }
        for (const [key, id] of Object.entries(ammoInputs)) if (currentReviewTab === 'ammo' && !sectionLocked('ammo') && $(id).value === '' && result.screen_state.ammo[key] !== null) { $(id).value = result.screen_state.ammo[key]; count++; $('ammoReviewed').checked = false; }
        for (const key of combatFields) if (currentReviewTab === 'combat_stats' && !sectionLocked('combat_stats') && $(key).value === '' && result.screen_state.combat_stats[key] !== null) { $(key).value = result.screen_state.combat_stats[key]; count++; $('combatReviewed').checked = false; }
        if (currentReviewTab === 'soul' && !sectionLocked('soul') && !$('soulReviewed').checked && result.screen_state.souls !== null) { renderSoulSlots(result.screen_state.souls); $('soulReviewed').checked = false; count++; }
        if (count) dirty = true;
        status(count ? 'Filled ' + count + ' fields. Review predictions before saving.' : 'No confident matches. Add reviewed examples with known values first.');
    } catch (error) { status('Prediction failed — ' + error.message, true); }
    finally { syncReviewLocks(); }
};

function switchReviewTab(name, focus = false, select = true) {
    if (name === 'screen') name = 'screen_kind';
    currentReviewTab = reviewTasks.some(task => task.key === name) ? name : 'pieces';
    const pieces = currentReviewTab === 'pieces';
    for (const task of reviewTasks) {
        const active = task.key === currentReviewTab;
        $(task.tab).setAttribute('aria-selected', String(active));
        $(task.tab).tabIndex = active ? 0 : -1;
        if (active && focus) $(task.tab).focus();
    }
    $('piecesView').hidden = !pieces; $('piecesControls').hidden = !pieces;
    $('screenView').hidden = pieces; $('screenControls').hidden = pieces;
    $('screenView').setAttribute('aria-labelledby', currentTask().tab);
    for (const fields of document.querySelectorAll('[data-review-task]')) fields.hidden = fields.dataset.reviewTask !== currentReviewTab;
    $('taskTitle').textContent = currentTask().label;
    $('draft').textContent = pieces ? 'Save draft' : 'Save progress';
    $('approve').textContent = pieces ? 'Approve pieces ✓' : 'Complete ✓';
    $('saveHelp').textContent = pendingPieceApprovalId ? 'Enter the floor to finish piece approval.' : '';
    if (items.length) {
        if (select && !loading && !dirty && !pendingPieceApprovalId && !saving) selectVisible();
        renderOptions(); showReviewStatus();
    }
}
function syncScreenKind() {
    if (items.length) renderOptions();
}
async function chooseTask(key, focus = false) {
    if (saving || key === currentReviewTab) return;
    const discard = dirty;
    if (discard && !confirm('Discard unsaved edits?')) return;
    dirty = false; editingSections.clear(); pendingPieceApprovalId = null;
    switchReviewTab(key, focus, false);
    await selectVisible(discard);
    history.replaceState(null, '', '#' + key);
}
for (const task of reviewTasks) {
    $(task.tab).onclick = () => chooseTask(task.key);
    $(task.tab).onkeydown = event => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const index = reviewTasks.indexOf(task);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? reviewTasks.length - 1 : (index + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1) + reviewTasks.length) % reviewTasks.length;
        chooseTask(reviewTasks[next].key, true);
    };
}
const initialTask = location.hash.slice(1);
switchReviewTab(initialTask === 'page-info' ? 'screen_kind' : initialTask || 'pieces');
