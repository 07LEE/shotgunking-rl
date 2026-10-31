(async()=>{
 const checks=[];
 function check(ok,name){if(!ok)throw Error(name);checks.push(name)}
 async function idle(){for(let i=0;i<300;i++){if(data&&!loading&&loadedId===currentId)return;await new Promise(r=>setTimeout(r,20))}throw Error('load timeout')}
 function queueFor(session){return Array.from(document.querySelectorAll('.queue-session')).find(group=>group.dataset.session===session)}
 async function pick(id){const button=Array.from(document.querySelectorAll('.queue-item')).find(button=>button.dataset.id===id);if(!button)throw Error('Missing queue item '+id);const group=button.closest('details');if(!group.open)group.querySelector('summary').click();button.click();await idle()}
 window.confirm=()=>true;
 try {
  await idle();
  check(currentId.endsWith('b-partial.png.json'),'starts at unfinished screenshot');
  check(document.querySelector('.review-sidebar').getBoundingClientRect().right<document.querySelector('.review-content').getBoundingClientRect().left,'review queue is left of the selected content');
  check(!document.querySelector('select#sessions, select#items'),'session and screenshot dropdowns removed');
  const previousId=currentId;queueFor('missing-floor').querySelector('summary').click();
  check(currentId===previousId,'expanding another session does not change the selected screenshot');
  check(queueFor('play').querySelectorAll('.queue-item').length===2&&!document.querySelector('[data-id="sessions/play/annotations/a-complete.png.json"]'),'completed screenshot hidden');
  switchReviewTab('screen');await idle();
  check($('ammo_loaded').disabled&&$('ammoReviewed').disabled,'reviewed ammunition locked');
  check(!$('attack').disabled&&!$('combatReviewed').disabled,'unfinished combat editable');
  check($('screenKind').disabled,'reviewed screen kind locked');
  const originalApi=api;
  api=async(url,...args)=>url==='/api/predict'?{floor:null,screen_state:{screen_kind:null,ammo:{loaded:9,reserve:9},combat_stats:{...Object.fromEntries(combatFields.map(key=>[key,null])),range_min:3},souls:null}}:originalApi(url,...args);
  await $('predictState').onclick();api=originalApi;
  check($('ammo_loaded').value==='2'&&$('ammoReviewed').checked&&$('range_min').value==='3'&&$('attack').value==='4','auto-fill leaves reviewed values and approvals intact');
  window.confirm=()=>false;document.querySelector('[data-id="sessions/play/annotations/c-pending.png.json"]').click();
  check(currentId===previousId&&$('range_min').value==='3','canceling sidebar navigation preserves unsaved edits');window.confirm=()=>true;
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  check(queueFor('play').querySelectorAll('.queue-item').length===3,'explicit completed view');
  await pick('sessions/play/annotations/a-complete.png.json');
  check($('ammo_loaded').disabled&&$('attack').disabled&&$('addSoul').disabled,'completed page sections locked');
  check($('draft').disabled&&$('approve').disabled,'completed page save actions locked');
  const oldRevision=revision;await save(false);check(revision===oldRevision,'completed page cannot be saved accidentally');
  switchReviewTab('pieces');await idle();
  check($('label').disabled&&Array.from($('grid').children).every(x=>x.disabled),'approved board locked');
  switchReviewTab('screen');await idle();
  document.querySelector('[aria-label="Edit reviewed ammunition"]').click();
  check(!$('ammo_loaded').disabled&&!$('ammoReviewed').checked&&dirty,'explicit edit reopens only ammo review');
  check($('attack').disabled&&$('screenKind').disabled,'other reviewed sections stay locked');
  $('ammo_loaded').value='1';$('ammo_loaded').oninput();await save(false);
  let stored=await api('/api/item?id='+encodeURIComponent(currentId));
  check(stored.data.confirmed&&stored.data.screen_state.reviews.screen_kind&&!stored.data.screen_state.reviews.ammo&&stored.data.screen_state.ammo.loaded===1,'editing ammo preserves board and unrelated approvals');
  $('ammoReviewed').checked=true;$('ammoReviewed').oninput();await save(false);
  check($('ammo_loaded').disabled&&$('draft').disabled,'review completion relocks edited section');
  $('showCompleted').checked=false;await $('showCompleted').onchange();await idle();
  check(!document.querySelector('[data-id="sessions/play/annotations/a-complete.png.json"]'),'finished screenshot leaves pending queue');
  check(!queueFor('finished'),'completed session hidden from pending queue');
  check(document.querySelector('.queue-item[aria-current="true"]').dataset.id===currentId,'sidebar highlights the selected screenshot');
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  await pick('sessions/finished/annotations/done.png.json');
  check(!document.querySelector('main').hidden&&$('ammo_loaded').disabled,'completed session can be inspected read-only');
  await pick('sessions/play/annotations/b-partial.png.json');
  switchReviewTab('pieces');await idle();await pick('sessions/play/annotations/c-pending.png.json');
  await save(true);await idle();
  check(currentId.endsWith('b-partial.png.json'),'next pending wraps to earlier unfinished image');
  $('showCompleted').checked=false;await $('showCompleted').onchange();await idle();
  await pick('sessions/missing-floor/annotations/no-floor.png.json');
  switchReviewTab('pieces');await idle();await save(true);
  check(currentReviewTab==='screen'&&!$('floor').disabled&&!$('approve').disabled,'missing floor can be supplied despite reviewed page sections');
  $('floor').value='2';$('floor').oninput();await save(true);
  const floorSaved=await api('/api/item?id='+encodeURIComponent('sessions/missing-floor/annotations/no-floor.png.json'));
  check(floorSaved.data.floor===2&&floorSaved.data.confirmed,'floor hand-off completes piece approval');
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  switchReviewTab('screen');await idle();await pick('sessions/cards/annotations/card-screen.png.json');
  check($('screenKind').disabled&&$('draft').disabled&&$('approve').disabled,'reviewed non-gameplay screenshot is read-only');
  check(!$('piecesTab').disabled,'completed non-gameplay screen does not trap category navigation');
  switchReviewTab('pieces');await idle();
  check(currentId.endsWith('b-partial.png.json'),'category navigation reaches another session with unfinished pieces');
  document.body.innerHTML='<pre id="browser-test-result">'+JSON.stringify({ok:true,checks})+'</pre>';
 }catch(e){document.body.innerHTML='<pre id="browser-test-result">'+JSON.stringify({ok:false,checks,error:e.stack})+'</pre>'}
})();
