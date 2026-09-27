import { $, bars, fail, int, kpis, legend, load, num, pct, seg, table, xy } from './kit.js';

try {
  const { summary: S, prices: P } = await load();
  const bge = S.students.bge, tf = S.students.tfidf;
  kpis($('#kpis'), [
    { label: 'Teacher accuracy', value: pct(S.teacher.accuracy), note: `GPT-3.5 Turbo, ${int(S.teacher.prompt_tokens)} prompt tokens a request` },
    { label: 'Embedding student', value: pct(bge.accuracy), note: `${pct(bge.agreement)} agreement with the teacher` },
    { label: 'Student latency', value: `${num(bge.speed.p50_ms, 0)} ms`, note: `median on ${S.threads} CPU threads; TF-IDF ${num(tf.speed.p50_ms, 1)} ms` },
    { label: 'Break-even', value: `${int(bge.economics.break_even_per_month)}`, note: 'requests a month, at the default prices' },
  ]);
  bars($('#acc'), [
    { label: 'Teacher · test', value: S.teacher.accuracy, text: pct(S.teacher.accuracy), color: 'var(--c6)' },
    { label: 'Teacher · variants', value: S.teacher.accuracy_variants, text: pct(S.teacher.accuracy_variants), color: 'var(--c6)', dim: true },
    { label: 'bge-small · test', value: bge.accuracy, text: pct(bge.accuracy) },
    { label: 'bge-small · variants', value: bge.accuracy_variants, text: pct(bge.accuracy_variants), dim: true },
    { label: 'TF-IDF · test', value: tf.accuracy, text: pct(tf.accuracy), color: 'var(--c2)' },
    { label: 'TF-IDF · variants', value: tf.accuracy_variants, text: pct(tf.accuracy_variants), color: 'var(--c2)', dim: true },
  ], { max: 1 });
  const lc = S.learning_curve;
  const cs = [{ name: 'accuracy', color: 'var(--accent)', dots: true, points: lc.map((r) => ({ x: r.reviews, y: r.accuracy, title: `${r.reviews} reviews (${r.examples} with variants): ${pct(r.accuracy)}` })) },
    { name: 'agreement with the teacher', color: 'var(--c2)', dash: true, points: lc.map((r) => ({ x: r.reviews, y: r.agreement })) }];
  xy($('#curve'), { label: 'Accuracy by training reviews', height: 220, series: cs, hline: { y: S.teacher.accuracy, label: 'teacher' }, x: { log: true, label: 'training reviews (log scale)', fmt: String, ticks: lc.map((r) => r.reviews) }, y: { min: 0.85, max: 0.97, fmt: (v) => `${Math.round(v * 100)}%` } });
  legend($('#curveKey'), cs);

  // distill/pipeline.py economics(), with the inputs from the sliders
  const labelled = S.train, host = P.student_host;
  const monthly = (student, perReq, perHour) => {
    const oneOff = (labelled * perReq) / P.amortise_months;
    return (v) => Math.max(1, Math.ceil(((v / (30 * 86400)) * P.peak_to_average) / student.speed.per_second)) * perHour * 730 + oneOff;
  };
  const vol = $('#vol'), tp = $('#tp'), hp = $('#hp');
  tp.value = Math.log10(1000 * S.teacher.cost_per_request).toFixed(3);
  hp.value = host.per_hour;
  let who = 'bge';
  const money = (x) => (x >= 100 ? `$${int(x)}` : `$${x.toFixed(2)}`);
  function draw() {
    const v = 10 ** +vol.value, perReq = tp.dataset.moved ? 10 ** +tp.value / 1000 : S.teacher.cost_per_request, perHour = hp.dataset.moved ? +hp.value : host.per_hour, st = S.students[who], f = monthly(st, perReq, perHour);
    $('#volVal').textContent = int(v);
    $('#tpVal').textContent = `$${(perReq * 1000).toFixed(2)}`;
    $('#hpVal').textContent = `$${perHour.toFixed(perHour < 0.1 ? 4 : 3)}`;
    let even = null;
    for (let i = 0; i <= 6000; i++) { const x = Math.floor(10 ** (3 + i / 1000)); if (f(x) <= x * perReq) { even = x; break; } }   // the bench's grid
    const machines = Math.max(1, Math.ceil(((v / (30 * 86400)) * P.peak_to_average) / st.speed.per_second));
    $('#costs').innerHTML = `<div>Teacher<b>${money(v * perReq)}</b><span class="muted small">a month</span></div>` +
      `<div>Student (${st.name.split(' +')[0]})<b>${money(f(v))}</b><span class="muted small">${machines} machine${machines > 1 ? 's' : ''} at ${int(st.speed.per_second)} requests/s each</span></div>` +
      `<div>Owning pays above<b>${even ? int(even) : 'never below 1B'}</b><span class="muted small">requests a month</span></div>`;
    const grid = []; for (let e = 3; e <= 9.0001; e += 0.05) grid.push(10 ** e);
    const series = [
      { name: 'teacher (API)', color: 'var(--c6)', points: grid.map((x) => ({ x, y: x * perReq })) },
      { name: `student: ${st.name}`, color: 'var(--accent)', points: grid.map((x) => ({ x, y: f(x) })) },
      { name: 'this volume', color: 'var(--text)', line: false, dots: true, points: [{ x: v, y: f(v), r: 5 }, { x: v, y: v * perReq, r: 5 }] },
    ];
    xy($('#costChart'), { label: 'Monthly cost by volume', height: 260, series, x: { log: true, min: 1000, max: 1e9, label: 'requests a month (log scale)', fmt: (x) => (x >= 1e9 ? '1B' : x >= 1e6 ? `${x / 1e6}M` : x >= 1e3 ? `${x / 1e3}k` : String(x)), ticks: [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9] },
      y: { log: true, min: 1, max: 1e7, fmt: (x) => (x >= 1e6 ? `$${x / 1e6}M` : x >= 1e3 ? `$${x / 1e3}k` : `$${x}`), ticks: [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7] },
      vline: even ? { x: even, label: 'break-even' } : null });
    legend($('#costKey'), series.slice(0, 2));
  }
  [vol, tp, hp].forEach((el) => { el.oninput = () => { el.dataset.moved = '1'; draw(); }; });   // the bench's exact prices until a slider moves
  seg($('#who'), [['bge', 'bge-small (94.9%)'], ['tfidf', 'TF-IDF (85.1%)']], who, (x) => { who = x; draw(); });
} catch (err) {
  fail(err);
}
