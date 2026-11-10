/* SicklePredict Watch Simulator — app logic (vanilla JS, no dependencies) */
(() => {
  const DATA = window.SICKLEPREDICT_DATA;
  const THR = DATA.threshold;
  const T = {
    ar: {
      eyebrow: 'SicklePredict · نموذج تفاعلي', title: 'محاكي ساعة SicklePredict',
      lede: 'شغّل سيناريو وشاهد ما تعرضه الساعة للمريض ساعة بساعة. درجة الخطر هي مخرجات نموذج Bi-LSTM المدرَّب نفسه على بيانات مرضى الاختبار، وليست أرقامًا تمثيلية.',
      notice: 'المرضى والإشارات محاكاة. يعرض هذا النموذج تجربة الاستخدام وسلوك الخوارزمية، ولا يثبت دقة سريرية.',
      prevScreen: '‹ الشاشة السابقة', nextScreen: 'الشاشة التالية ›',
      phoneApp: 'تطبيق الفريق الطبي', phoneTitle: 'تنبيه خطر مبكر لمريض',
      pickTitle: 'اختر سيناريو', restart: 'من البداية', speed: 'السرعة', speedSlow: 'بطيء', speedMed: 'متوسط', speedFast: 'سريع',
      pauseOnAlert: 'إيقاف مؤقت عند التنبيه', play: '▶ تشغيل', pause: '❚❚ إيقاف',
      lgRisk: 'درجة الخطر', lgThr: 'عتبة التنبيه', lgVoc: 'نوبة', lgCold: 'تعرّض للبرد', lgPro: 'العلامة السابقة للنوبة (مخفية عن الساعة)',
      truthTitle: 'ما يحدث فعلًا داخل جسم المريض', truthHint: 'هذه هي «الحقيقة» داخل المحاكاة. الساعة لا تراها، وترى فقط الإشارات.',
      sigTitle: 'الإشارة التي تقرؤها الساعة', sigHint: 'الخط الرفيع: القيمة كل ساعة. الخط العريض: متوسط ٦ ساعات. المنطقة الحمراء: النوبة.',
      logTitle: 'سجل الأحداث', logEmpty: 'لا أحداث بعد. اضغط تشغيل.',
      scn: {
        caught_cold: ['نوبة بعد التعرض للبرد', 'تنبيه مبكر قبل النوبة بـ ٢٠ ساعة'],
        caught_heat: ['نوبة بعد يوم حار', 'تنبيه قبل النوبة بـ ١٢ ساعة'],
        false_alarm: ['إنذار كاذب', 'انقباض وعائي مؤقت لم يتطور إلى نوبة'],
        missed: ['نوبة لم تُكتشف', 'حالة لم يلتقطها النموذج، للأمانة'],
        calm: ['أسبوع هادئ', 'تعرّض متكرر للبرد بدون إنذارات'],
      },
      level: { low: 'منخفض', mod: 'متوسط', high: 'مرتفع', none: '—' },
      riskSub: 'خطر نوبة خلال ٢٤ ساعة',
      v: { hr: 'نبض', spo2: 'أكسجين', skin: 'حرارة', hrv: 'تغيّر النبض', pi: 'تروية ٨١٠ نانومتر', eda: 'توصيل الجلد', transit: 'زمن العبور', steps: 'خطوات/ساعة' },
      vitalsTitle: 'العلامات الحيوية', trendTitle: 'الخطر آخر ٢٤ ساعة', trendNote: 'الخط المتقطع = عتبة التنبيه',
      devNote: 'مقارنة بخط المريض الأساسي',
      alertTitle: 'تنبيه مبكر: خطر نوبة ألم',
      steps: ['💧 اشرب ٥٠٠ مل ماء الآن', '🧥 ابقَ دافئًا وتجنّب المكيف', '🛌 خفّف النشاط وارتح', '📞 تواصل مع فريقك الطبي'],
      ok: 'تم', call: 'اتصال',
      offTitle: 'السوار غير ملبوس', offText: 'ارتدِ السوار لاستئناف المراقبة',
      nodataTitle: 'جمع البيانات', nodataText: 'لا توجد بيانات كافية في آخر ٢٤ ساعة',
      vocTitle: 'تم تسجيل نوبة ألم', vocText: 'اتبع خطة الألم المتفق عليها، وتوجّه للطوارئ إذا اشتد الألم أو ظهرت حرارة أو ضيق نفس.', vocBtn: 'خطة الألم',
      callingTitle: 'جارٍ الاتصال…', callingText: 'عيادة أمراض الدم – خط المتابعة', end: 'إنهاء',
      f: { cold: 'تعرّض للبرد', pro: 'علامة ما قبل النوبة', ab: 'انقباض مؤقت', inf: 'التهاب', voc: 'نوبة جارية', sleep: 'نائم' },
      cdBefore: (h) => `النوبة ستبدأ بعد <b>${h}</b> ساعة. الساعة لا تعرف ذلك.`, cdNone: 'لا نوبة قادمة في هذا السيناريو.', cdIn: 'النوبة جارية الآن.',
      narr: {
        quiet: 'كل شيء ضمن خط المريض الأساسي. الساعة تراقب بصمت.',
        sleep: 'المريض نائم. التروية الطرفية وحرارة الجلد ترتفع طبيعيًا أثناء النوم، والنموذج يقارن بقيم المريض الليلية.',
        cold: 'المريض تعرّض للبرد. تنخفض التروية وحرارة الجلد، لكن توصيل الجلد وزمن العبور لا يتغيران، فلا يرفع النموذج الخطر كثيرًا.',
        pro: 'بدأت تغيّرات ما قبل النوبة: انقباض وعائي، وزيادة زمن العبور، وارتفاع توصيل الجلد. لم يشعر المريض بأي ألم بعد.',
        ab: 'انقباض وعائي مؤقت يشبه بداية نوبة، لكنه سيتراجع من تلقاء نفسه.',
        inf: 'التهاب عابر يرفع النبض والحرارة ويزيد قابلية حدوث نوبة.',
        voc: 'النوبة بدأت. في هذه الفترة لا يحسب النموذج خطرًا جديدًا، والتطبيق يوجّه المريض لخطة الألم.',
        off: 'السوار مخلوع (شحن أو نسيان). لا توجد إشارات في هذه الساعة.',
      },
      lg: { alert: 'تنبيه أُرسل للمريض والفريق الطبي', onset: 'بداية نوبة ألم', ok: 'المريض أكّد استلام التنبيه', call: 'المريض اتصل بالفريق الطبي', lvl: (l) => `تغيّر مستوى الخطر إلى ${l}` },
      phoneBody: (pid, r) => `${pid}: درجة الخطر ${r}%. يُقترح التواصل مع المريض خلال ساعتين.`,
      out: {
        caught_cold: 'النتيجة: أطلقت الساعة أول تنبيه قبل النوبة بـ ٢٠ ساعة، وهذا وقت كافٍ للترطيب والتدفئة والاستشارة.',
        caught_heat: 'النتيجة: تنبيه قبل النوبة بـ ١٢ ساعة.',
        false_alarm: 'النتيجة: تنبيه واحد لم تتبعه نوبة. هذا ثمن الحساسية العالية، ونسبته في النموذج حوالي ٠٫١٣ إنذار كاذب لكل مريض أسبوعيًا.',
        missed: 'النتيجة: لم تكن العلامة واضحة عند هذا المريض، فبقي الخطر أقل من العتبة. النموذج التقط ٣٣ نوبة من ٣٩ في الاختبار.',
        calm: 'النتيجة: رغم التعرض المتكرر للبرد، لم يُطلق أي إنذار.',
      },
      sexF: 'أنثى', sexM: 'ذكر', yrs: 'سنة', hu: 'هيدروكسي يوريا',
      days: ['الأحد','الاثنين','الثلاثاء','الأربعاء','الخميس','الجمعة','السبت'],
    },
    en: {
      eyebrow: 'SicklePredict · interactive prototype', title: 'SicklePredict Watch Simulator',
      lede: 'Play a scenario and watch what the wristband shows the patient, hour by hour. The risk score is the trained Bi-LSTM model\'s own output on test-set patients, not illustrative numbers.',
      notice: 'Patients and signals are simulated. This demonstrates the user experience and algorithm behaviour, not clinical accuracy.',
      prevScreen: '‹ Previous screen', nextScreen: 'Next screen ›',
      phoneApp: 'Care-team app', phoneTitle: 'Early-risk alert for a patient',
      pickTitle: 'Choose a scenario', restart: 'Restart', speed: 'Speed', speedSlow: 'Slow', speedMed: 'Medium', speedFast: 'Fast',
      pauseOnAlert: 'Pause on alert', play: '▶ Play', pause: '❚❚ Pause',
      lgRisk: 'Risk score', lgThr: 'Alert threshold', lgVoc: 'Crisis', lgCold: 'Cold exposure', lgPro: 'Pre-crisis signature (hidden from the watch)',
      truthTitle: 'What is really happening in the body', truthHint: 'This is the simulation\'s ground truth. The watch cannot see it; it only sees the signals.',
      sigTitle: 'The signal the watch reads', sigHint: 'Thin line: hourly value. Thick line: 6-hour average. Red band: crisis.',
      logTitle: 'Event log', logEmpty: 'Nothing yet. Press Play.',
      scn: {
        caught_cold: ['Crisis after cold exposure', 'Early alert 20 h before onset'],
        caught_heat: ['Crisis after a hot day', 'Alert 12 h before onset'],
        false_alarm: ['False alarm', 'Transient vasoconstriction that resolved'],
        missed: ['Missed crisis', 'A case the model did not catch'],
        calm: ['Quiet week', 'Repeated cold exposure, no alarms'],
      },
      level: { low: 'Low', mod: 'Moderate', high: 'High', none: '—' },
      riskSub: 'Crisis risk, next 24 h',
      v: { hr: 'Heart', spo2: 'SpO₂', skin: 'Skin', hrv: 'HRV', pi: 'Perfusion 810 nm', eda: 'Skin conductance', transit: 'Transit delay', steps: 'Steps/hour' },
      vitalsTitle: 'Vitals', trendTitle: 'Risk, last 24 h', trendNote: 'Dashed line = alert threshold',
      devNote: 'vs. personal baseline',
      alertTitle: 'Early warning: pain-crisis risk',
      steps: ['💧 Drink 500 ml of water now', '🧥 Keep warm, avoid cold A/C', '🛌 Reduce activity and rest', '📞 Contact your care team'],
      ok: 'Got it', call: 'Call',
      offTitle: 'Band not worn', offText: 'Put the band back on to resume monitoring',
      nodataTitle: 'Collecting data', nodataText: 'Not enough data in the last 24 h',
      vocTitle: 'Pain crisis logged', vocText: 'Follow your agreed pain plan. Go to the ED if pain is severe or you have fever or shortness of breath.', vocBtn: 'Pain plan',
      callingTitle: 'Calling…', callingText: 'Haematology clinic – follow-up line', end: 'End',
      f: { cold: 'Cold exposure', pro: 'Pre-crisis signature', ab: 'Transient vasoconstriction', inf: 'Infection', voc: 'Crisis ongoing', sleep: 'Asleep' },
      cdBefore: (h) => `Crisis starts in <b>${h}</b> h. The watch does not know this.`, cdNone: 'No crisis coming in this scenario.', cdIn: 'Crisis is ongoing.',
      narr: {
        quiet: 'Everything is within the patient\'s own baseline. The watch monitors silently.',
        sleep: 'Patient asleep. Peripheral perfusion and skin temperature rise normally at night; the model compares with this patient\'s night-time values.',
        cold: 'Cold exposure. Perfusion and skin temperature drop, but skin conductance and transit delay do not, so the model keeps the risk low.',
        pro: 'Pre-crisis changes have started: vasoconstriction, longer transit delay and rising skin conductance. The patient feels no pain yet.',
        ab: 'A transient vasoconstriction that looks like a prodrome, but will resolve on its own.',
        inf: 'An intercurrent infection raises heart rate and temperature and makes a crisis more likely.',
        voc: 'The crisis has started. The model stops scoring new risk; the app points the patient to the pain plan.',
        off: 'Band off (charging or forgotten). No signals this hour.',
      },
      lg: { alert: 'Alert sent to patient and care team', onset: 'Pain-crisis onset', ok: 'Patient acknowledged the alert', call: 'Patient called the care team', lvl: (l) => `Risk level changed to ${l}` },
      phoneBody: (pid, r) => `${pid}: risk ${r}%. Suggest contacting the patient within 2 h.`,
      out: {
        caught_cold: 'Outcome: first alert 20 h before onset, enough time for fluids, warmth and a tele-consult.',
        caught_heat: 'Outcome: alert 12 h before onset.',
        false_alarm: 'Outcome: one alert with no crisis. That is the cost of sensitivity; the model averages about 0.13 false alerts per patient-week.',
        missed: 'Outcome: this patient\'s signature was faint and risk stayed under the threshold. The model caught 33 of 39 test crises.',
        calm: 'Outcome: despite repeated cold exposure, no alarm fired.',
      },
      sexF: 'Female', sexM: 'Male', yrs: 'y', hu: 'hydroxyurea',
      days: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],
    },
  };
  const ARD = '٠١٢٣٤٥٦٧٨٩';
  let lang = 'ar';
  const L = () => T[lang];
  const nf = (x, d = 0) => { if (x === null || x === undefined || Number.isNaN(x)) return '—'; const s = Number(x).toFixed(d); return lang === 'ar' ? s.replace(/\d/g, (c) => ARD[c]).replace('.', '٫') : s; };

  // ---------- state ----------
  let phoneK = null, S = DATA.scenarios[0], i = 0, playing = false, timer = null, screenIdx = 0, overlay = null, logItems = [], lastLevel = null;
  const SCREENS = ['home', 'vitals', 'trend'];
  const $ = (id) => document.getElementById(id);

  // personal baseline proxy for the "vs baseline" arrows: median of the scenario's first 48 non-crisis hours
  const baseOf = (key) => { const v = S[key].slice(0, 48).filter((x, k) => x !== null && !S.voc[k]); v.sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; };
  let BASE = {};
  const level = (r) => r === null ? 'none' : r >= THR ? 'high' : r >= THR / 2 ? 'mod' : 'low';
  const lvlColor = { low: 'var(--w-low)', mod: 'var(--w-mod)', high: 'var(--w-high)', none: '#555' };
  const parseTs = (s) => new Date(s.replace(' ', 'T') + ':00Z');
  const tsLabel = (k) => { const d = parseTs(S.ts[k]); const hh = String(d.getUTCHours()).padStart(2, '0'); return `${L().days[d.getUTCDay()]} ${lang === 'ar' ? nf(hh) : hh}:${lang === 'ar' ? '٠٠' : '00'}`; };
  const shortTime = (k) => { const hh = String(parseTs(S.ts[k]).getUTCHours()).padStart(2, '0'); return lang === 'ar' ? `${nf(hh)}:٠٠` : `${hh}:00`; };

  // ---------- i18n ----------
  function applyLang() {
    document.documentElement.lang = lang;
    $('app').dir = lang === 'ar' ? 'rtl' : 'ltr';
    document.querySelectorAll('[data-t]').forEach((el) => { const v = L()[el.dataset.t]; if (typeof v === 'string') el.textContent = v; });
    $('lang-ar').setAttribute('aria-pressed', lang === 'ar'); $('lang-en').setAttribute('aria-pressed', lang === 'en');
    buildScenarios(); buildSigTabs(); renderAll();
  }
  $('lang-ar').onclick = () => { lang = 'ar'; applyLang(); };
  $('lang-en').onclick = () => { lang = 'en'; applyLang(); };

  // ---------- scenario picker ----------
  function buildScenarios() {
    const box = $('scns'); box.replaceChildren();
    for (const sc of DATA.scenarios) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'scn';
      b.setAttribute('aria-pressed', sc.id === S.id);
      const p = sc.profile;
      const [t, d] = L().scn[sc.id];
      const bt = document.createElement('b'); bt.textContent = t;
      const sd = document.createElement('span'); sd.textContent = d;
      const sp = document.createElement('span'); sp.textContent = `${sc.patient} · ${p.genotype} · ${p.sex === 'F' ? L().sexF : L().sexM} ${nf(p.age)} ${L().yrs}`;
      b.append(bt, sd, sp);
      b.onclick = () => loadScenario(sc);
      box.appendChild(b);
    }
  }
  function loadScenario(sc) {
    stop(); S = sc; i = 0; overlay = null; screenIdx = 0; logItems = []; lastLevel = null;
    BASE = Object.fromEntries(['hr', 'spo2', 'skin', 'hrv', 'pi', 'eda', 'transit', 'steps'].map((k) => [k, baseOf(k)]));
    $('scrub').max = S.ts.length - 1; $('scrub').value = 0;
    $('phone-note').hidden = true; $('outcome').hidden = true;
    buildScenarios(); renderAll();
  }

  // ---------- playback ----------
  function setPlayLabel() { $('play').textContent = playing ? L().pause : L().play; }
  function stop() { playing = false; clearInterval(timer); timer = null; setPlayLabel(); }
  function start() {
    if (i >= S.ts.length - 1) { loadScenario(S); }
    playing = true; setPlayLabel();
    timer = setInterval(step, +$('speed').value);
  }
  $('play').onclick = () => playing ? stop() : start();
  $('restart').onclick = () => loadScenario(S);
  $('speed').onchange = () => { if (playing) { clearInterval(timer); timer = setInterval(step, +$('speed').value); } };
  $('scrub').oninput = () => { stop(); i = +$('scrub').value; overlay = null; rebuildLog(); renderAll(); };

  function step() {
    if (i >= S.ts.length - 1) { stop(); showOutcome(); return; }
    i++; $('scrub').value = i;
    handleEvents(i, true);
    renderAll();
    if (i >= S.ts.length - 1) { stop(); showOutcome(); }
  }
  function handleEvents(k, live) {
    const lv = S.voc[k] ? null : level(S.risk[k]);
    if (S.alert[k]) {
      logItems.push({ k, kind: 'alert', text: 'alert' });
      if (live) { overlay = 'alert'; buzz(); showPhone(k); if ($('pause-alert').checked) stop(); }
    }
    if (S.onset[k]) { logItems.push({ k, kind: 'voc', text: 'onset' }); if (live) overlay = null; }
    if (lv && lv !== 'none' && lastLevel && lv !== lastLevel) logItems.push({ k, kind: 'lvl', lvl: lv });
    if (lv && lv !== 'none') lastLevel = lv;
  }
  function rebuildLog() { logItems = []; lastLevel = null; for (let k = 0; k <= i; k++) handleEvents(k, false); $('phone-note').hidden = true; $('outcome').hidden = i < S.ts.length - 1; if (i >= S.ts.length - 1) showOutcome(); }
  function buzz() { const c = $('case'); c.classList.remove('buzz'); void c.offsetWidth; c.classList.add('buzz'); }
  function showPhone(k) {
    phoneK = k; $('phone-note').hidden = false;
    $('phone-time').textContent = shortTime(k);
    $('phone-body').textContent = L().phoneBody(S.patient, nf(S.risk[k] * 100));
  }
  function showOutcome() { $('outcome').hidden = false; $('outcome').textContent = L().out[S.id]; }

  // ---------- watch ----------
  const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt !== undefined) e.textContent = txt; return e; };
  function statusBar() {
    const s = el('div', 'w-status');
    s.append(el('span', 'app', 'SicklePredict'), el('span', 'w-clock', shortTime(i)));
    return s;
  }
  function dots() { const d = el('div', 'w-dots'); SCREENS.forEach((_, k) => { const x = el('i'); if (k === screenIdx) x.className = 'on'; d.appendChild(x); }); return d; }
  function ring(r) {
    const lv = level(r), C = 2 * Math.PI * 62, frac = r === null ? 0 : Math.max(0.02, Math.min(1, r));
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', '0 0 150 150');
    const bg = document.createElementNS(NS, 'circle'); Object.entries({ cx: 75, cy: 75, r: 62, fill: 'none', stroke: '#222025', 'stroke-width': 11 }).forEach(([a, b]) => bg.setAttribute(a, b));
    const fg = document.createElementNS(NS, 'circle'); Object.entries({ cx: 75, cy: 75, r: 62, fill: 'none', stroke: lvlColor[lv], 'stroke-width': 11, 'stroke-linecap': 'round', 'stroke-dasharray': `${frac * C} ${C}`, transform: 'rotate(-90 75 75)' }).forEach(([a, b]) => fg.setAttribute(a, b));
    fg.style.transition = 'stroke-dasharray .4s ease';
    const tk = document.createElementNS(NS, 'line'); const ang = -Math.PI / 2 + 2 * Math.PI * THR;
    Object.entries({ x1: 75 + 52 * Math.cos(ang), y1: 75 + 52 * Math.sin(ang), x2: 75 + 72 * Math.cos(ang), y2: 75 + 72 * Math.sin(ang), stroke: '#fff', 'stroke-width': 2, 'stroke-opacity': .7 }).forEach(([a, b]) => tk.setAttribute(a, b));
    svg.append(bg, fg, tk);
    return svg;
  }
  function vit(label, val, unit) { const d = el('div', 'w-vit'); const b = el('b', '', val); d.append(b, el('span', '', `${label}${unit ? ' ' + unit : ''}`)); return d; }
  function renderWatch() {
    const scr = $('screen'); scr.replaceChildren();
    const v = el('div', 'w-view');
    const worn = S.worn[i] === 1, r = S.risk[i], inVoc = S.voc[i] === 1;
    if (overlay === 'calling') {
      v.classList.add('w-center');
      v.style.display = 'grid';
      v.append(el('div', 'big', L().callingTitle), el('p', '', L().callingText));
      const b = el('button', 'w-btn call', L().end); b.type = 'button'; b.onclick = () => { overlay = null; renderWatch(); };
      v.appendChild(b); scr.appendChild(v); return;
    }
    if (overlay === 'alert') {
      v.classList.add('w-alert'); v.style.gridTemplateRows = 'auto 1fr auto';
      const top = el('div'); const ic = el('div', 'w-icon pulse', '!'); top.append(ic, el('h3', '', L().alertTitle));
      const ul = el('ul', 'w-steps'); L().steps.forEach((s) => ul.appendChild(el('li', '', s)));
      const mid = el('div'); mid.style.overflow = 'hidden'; mid.appendChild(ul);
      const btns = el('div', 'w-btns');
      const ok = el('button', 'w-btn primary', L().ok); ok.type = 'button';
      ok.onclick = () => { overlay = null; logItems.push({ k: i, kind: 'act', text: 'ok' }); renderAll(); };
      const call = el('button', 'w-btn call', L().call); call.type = 'button';
      call.onclick = () => { overlay = 'calling'; logItems.push({ k: i, kind: 'act', text: 'call' }); renderAll(); };
      btns.append(ok, call);
      v.append(top, mid, btns); scr.appendChild(v); return;
    }
    v.appendChild(statusBar());
    if (inVoc) {
      const c = el('div', 'w-center'); c.append(el('div', 'big', L().vocTitle), el('p', '', L().vocText));
      v.append(c, el('div'));
    } else if (!worn) {
      const c = el('div', 'w-center'); c.append(el('div', 'big', L().offTitle), el('p', '', L().offText));
      v.append(c, el('div'));
    } else if (SCREENS[screenIdx] === 'home') {
      const box = el('div', 'ring-box'); box.appendChild(ring(r));
      const cc = el('div', 'ring-center');
      const lv = level(r);
      if (r === null) { cc.append(el('div', 'ring-lbl', L().nodataTitle), el('div', 'ring-sub', L().nodataText)); }
      else {
        const val = el('div', 'ring-val'); val.append(nf(r * 100), el('small', '', '%'));
        const lb = el('div', 'ring-lbl', L().level[lv]); lb.style.color = lvlColor[lv];
        cc.append(val, lb, el('div', 'ring-sub', L().riskSub));
      }
      box.appendChild(cc);
      const vv = el('div', 'w-vitals');
      vv.append(vit('♥', nf(S.hr[i]), ''), vit('SpO₂', nf(S.spo2[i]), '%'), vit('°C', nf(S.skin[i], 1), ''));
      v.append(box, vv);
    } else if (SCREENS[screenIdx] === 'vitals') {
      const list = el('div', 'w-list');
      list.appendChild(el('div', 'w-title', L().vitalsTitle));
      const rows = [['hr', 0, 'bpm'], ['spo2', 0, '%'], ['skin', 1, '°C'], ['pi', 2, '%'], ['eda', 2, 'µS'], ['transit', 0, 'ms']];
      for (const [k, d, u] of rows) {
        const row = el('div', 'w-row'); const lab = el('span', '', L().v[k]);
        const val = el('span'); val.append(el('b', '', nf(S[k][i], d)), ` ${u}`);
        if (BASE[k] !== null && S[k][i] !== null) {
          const diff = S[k][i] - BASE[k], rel = Math.abs(diff) / (Math.abs(BASE[k]) || 1);
          const up = diff > 0, big = rel > (k === 'skin' ? 0.02 : k === 'spo2' ? 0.015 : 0.1);
          const dv = el('span', 'dev', big ? (up ? '▲' : '▼') : '•'); dv.style.color = big ? 'var(--w-mod)' : 'var(--w-muted)'; val.appendChild(dv);
        }
        row.append(lab, val); list.appendChild(row);
      }
      v.append(list);
    } else {
      const tw = el('div', 'w-trend'); tw.appendChild(el('div', 'w-title', L().trendTitle));
      tw.appendChild(miniTrend()); tw.appendChild(el('div', 'w-note', L().trendNote));
      v.append(tw);
    }
    v.appendChild(dots());
    scr.appendChild(v);
  }
  function miniTrend() {
    const NS = 'http://www.w3.org/2000/svg', W = 220, H = 120;
    const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const a = Math.max(0, i - 23), pts = [];
    for (let k = a; k <= i; k++) if (S.risk[k] !== null) pts.push([(k - a) / 23 * (W - 8) + 4, H - 6 - S.risk[k] * (H - 14)]);
    const thrY = H - 6 - THR * (H - 14);
    const ln = document.createElementNS(NS, 'line'); Object.entries({ x1: 4, x2: W - 4, y1: thrY, y2: thrY, stroke: '#888', 'stroke-dasharray': '4 3' }).forEach(([k, v]) => ln.setAttribute(k, v));
    svg.appendChild(ln);
    if (pts.length > 1) {
      const p = document.createElementNS(NS, 'path'); p.setAttribute('d', 'M' + pts.map((q) => q.join(',')).join('L'));
      p.setAttribute('fill', 'none'); p.setAttribute('stroke', lvlColor[level(S.risk[i])] || '#ccc'); p.setAttribute('stroke-width', 2.5); p.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(p);
      const last = pts[pts.length - 1]; const c = document.createElementNS(NS, 'circle'); c.setAttribute('cx', last[0]); c.setAttribute('cy', last[1]); c.setAttribute('r', 4); c.setAttribute('fill', '#fff'); svg.appendChild(c);
    }
    return svg;
  }
  const nextScreen = (d) => { if (overlay) return; screenIdx = (screenIdx + d + SCREENS.length) % SCREENS.length; renderWatch(); };
  $('crown').onclick = () => nextScreen(1);
  $('next-screen').onclick = () => nextScreen(lang === 'ar' ? 1 : 1);
  $('prev-screen').onclick = () => nextScreen(-1);

  // ---------- console charts ----------
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function bands(key, test) { const out = []; let s = null; S[key].forEach((x, k) => { const on = test(x); if (on && s === null) s = k; if ((!on || k === S[key].length - 1) && s !== null) { out.push([s, on ? k : k - 1]); s = null; } }); return out; }
  function drawTimeline() {
    const box = $('tl'); const W = Math.max(300, box.clientWidth || 600), H = 150, m = { l: 34, r: 8, t: 10, b: 22 };
    const n = S.ts.length, X = (k) => m.l + k / (n - 1) * (W - m.l - m.r), Y = (r) => H - m.b - r * (H - m.b - m.t);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="risk timeline">`;
    for (const [a, b] of bands('pro', (x) => x > 0.05)) s += `<rect x="${X(a)}" y="${m.t}" width="${Math.max(2, X(b) - X(a))}" height="${H - m.b - m.t}" fill="var(--warn)" fill-opacity=".18"/>`;
    for (const [a, b] of bands('ab', (x) => x > 0)) s += `<rect x="${X(a)}" y="${m.t}" width="${Math.max(2, X(b) - X(a))}" height="${H - m.b - m.t}" fill="var(--warn)" fill-opacity=".18"/>`;
    for (const [a, b] of bands('cold', (x) => x > 0)) s += `<rect x="${X(a)}" y="${H - m.b - 8}" width="${Math.max(2, X(b) - X(a))}" height="8" fill="var(--info)" fill-opacity=".45"/>`;
    for (const [a, b] of bands('voc', (x) => x > 0)) s += `<rect x="${X(a)}" y="${m.t}" width="${Math.max(2, X(b) - X(a))}" height="${H - m.b - m.t}" fill="var(--bad)" fill-opacity=".16"/>`;
    for (const t of [0, .25, .5, .75, 1]) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--line)"/><text x="${m.l - 6}" y="${Y(t)}" dy=".35em" text-anchor="end" font-size="10" fill="var(--muted)" font-family="var(--mono)">${nf(t * 100)}%</text>`;
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(THR)}" y2="${Y(THR)}" stroke="var(--muted)" stroke-dasharray="5 4"/>`;
    let d = '', pen = false;
    S.risk.forEach((r, k) => { if (r === null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(k).toFixed(1) + ',' + Y(r).toFixed(1); pen = true; });
    s += `<path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>`;
    S.alert.forEach((a, k) => { if (a) s += `<path d="M${X(k) - 5},${m.t - 2} L${X(k) + 5},${m.t - 2} L${X(k)},${m.t + 6} Z" fill="var(--bad)"/>`; });
    for (let k = 0; k < n; k += 24) s += `<text x="${X(k)}" y="${H - 6}" font-size="10" fill="var(--muted)" text-anchor="middle">${L().days[parseTs(S.ts[k]).getUTCDay()]}</text>`;
    s += `<line x1="${X(i)}" x2="${X(i)}" y1="${m.t - 4}" y2="${H - m.b}" stroke="var(--fg)" stroke-width="1.5"/>`;
    if (S.risk[i] !== null) s += `<circle cx="${X(i)}" cy="${Y(S.risk[i])}" r="4.5" fill="var(--accent)" stroke="var(--panel)" stroke-width="2"/>`;
    box.innerHTML = s + '</svg>';
  }
  let SIG = 'pi';
  const SIGS = { pi: [2, '%'], eda: [2, 'µS'], skin: [1, '°C'], transit: [0, 'ms'], hr: [0, 'bpm'], spo2: [0, '%'] };
  function buildSigTabs() {
    const box = $('sig-tabs'); box.replaceChildren();
    for (const k of Object.keys(SIGS)) {
      const b = el('button', 'chip-btn', L().v[k]); b.type = 'button'; b.setAttribute('aria-pressed', k === SIG);
      b.onclick = () => { SIG = k; buildSigTabs(); drawSignal(); };
      box.appendChild(b);
    }
  }
  function drawSignal() {
    const box = $('sig'); const W = Math.max(300, box.clientWidth || 600), H = 130, m = { l: 44, r: 8, t: 8, b: 18 };
    const vals = S[SIG], n = vals.length, fin = vals.filter((x) => x !== null);
    let lo = Math.min(...fin), hi = Math.max(...fin); const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
    const X = (k) => m.l + k / (n - 1) * (W - m.l - m.r), Y = (v) => H - m.b - (v - lo) / (hi - lo) * (H - m.b - m.t);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="signal">`;
    for (const [a, b] of bands('voc', (x) => x > 0)) s += `<rect x="${X(a)}" y="${m.t}" width="${Math.max(2, X(b) - X(a))}" height="${H - m.b - m.t}" fill="var(--bad)" fill-opacity=".14"/>`;
    for (const t of [0, .5, 1]) { const v = lo + t * (hi - lo); s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l - 6}" y="${Y(v)}" dy=".35em" text-anchor="end" font-size="10" fill="var(--muted)" font-family="var(--mono)">${nf(v, SIGS[SIG][0])}</text>`; }
    let d = '', pen = false; vals.forEach((v, k) => { if (v === null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(k).toFixed(1) + ',' + Y(v).toFixed(1); pen = true; });
    s += `<path d="${d}" fill="none" stroke="var(--muted)" stroke-opacity=".55" stroke-width="1"/>`;
    let d2 = '', pen2 = false; vals.forEach((v, k) => { const w = vals.slice(Math.max(0, k - 5), k + 1).filter((x) => x !== null); if (w.length < 3) { pen2 = false; return; } const mu = w.reduce((a, b) => a + b, 0) / w.length; d2 += (pen2 ? 'L' : 'M') + X(k).toFixed(1) + ',' + Y(mu).toFixed(1); pen2 = true; });
    s += `<path d="${d2}" fill="none" stroke="var(--info)" stroke-width="2.2" stroke-linejoin="round"/>`;
    s += `<line x1="${X(i)}" x2="${X(i)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--fg)" stroke-width="1.5"/>`;
    s += `<text x="${W - m.r}" y="${m.t + 10}" text-anchor="end" font-size="11" fill="var(--fg)" font-family="var(--mono)">${nf(vals[i], SIGS[SIG][0])} ${SIGS[SIG][1]}</text>`;
    box.innerHTML = s + '</svg>';
  }

  // ---------- ground truth + narration + log ----------
  function renderTruth() {
    const f = $('facts'); f.replaceChildren();
    const on = { cold: S.cold[i] > 0, pro: S.pro[i] > 0.05, ab: S.ab[i] > 0, inf: S.inf[i] > 0, voc: S.voc[i] > 0, sleep: S.sleep[i] > 0 };
    for (const k of Object.keys(on)) { const p = el('span', 'pill ' + k + (on[k] ? ' on' : ''), L().f[k] + (k === 'pro' && on.pro ? ` ${nf(S.pro[i] * 100)}%` : '')); f.appendChild(p); }
    const onsetK = S.onset.indexOf(1);
    const cd = $('countdown');
    if (S.voc[i]) cd.textContent = L().cdIn;
    else if (onsetK > i) cd.innerHTML = L().cdBefore(nf(onsetK - i));
    else cd.textContent = L().cdNone;
    const N = L().narr;
    $('narr').textContent = S.voc[i] ? N.voc : S.worn[i] === 0 ? N.off : on.pro ? N.pro : on.ab ? N.ab : on.inf ? N.inf : on.cold ? N.cold : on.sleep ? N.sleep : N.quiet;
  }
  function renderLog() {
    const ul = $('log'); ul.replaceChildren();
    if (!logItems.length) { ul.appendChild(el('li', 'log-empty', L().logEmpty)); return; }
    for (const it of [...logItems].reverse()) {
      const li = el('li'); const t = el('time', '', tsLabel(it.k));
      const txt = it.kind === 'lvl' ? L().lg.lvl(L().level[it.lvl]) : L().lg[it.text];
      const sp = el('span', it.kind === 'alert' ? 'k-alert' : it.kind === 'voc' ? 'k-voc' : it.kind === 'act' ? 'k-act' : '', txt);
      li.append(t, sp); ul.appendChild(li);
    }
  }
  function renderAll() {
    setPlayLabel();
    $('clock-big').textContent = `${S.patient} · ${tsLabel(i)}`;
    renderWatch(); drawTimeline(); drawSignal(); renderTruth(); renderLog();
    if (!$('outcome').hidden) $('outcome').textContent = L().out[S.id];
    if (!$('phone-note').hidden && phoneK !== null) { $('phone-time').textContent = shortTime(phoneK); $('phone-body').textContent = L().phoneBody(S.patient, nf(S.risk[phoneK] * 100)); }
  }
  window.addEventListener('resize', () => { drawTimeline(); drawSignal(); });

  loadScenario(DATA.scenarios[0]);
  applyLang();
})();
