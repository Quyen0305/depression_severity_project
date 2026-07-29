const API_BASE = 'http://127.0.0.1:5000';
let history = [];
let lastResult = null;

const encourageQuotes = [
  '💛 "Bạn đã dũng cảm khi ở đây và lắng nghe chính mình."',
  '🌤️ "Không sao nếu hôm nay chưa ổn, ngày mai vẫn còn cơ hội."',
  '🌱 "Mỗi bước nhỏ chăm sóc bản thân đều có ý nghĩa."',
  '☕ "Bạn xứng đáng được nghỉ ngơi, không cần lý do."'
];
document.getElementById('encourageText').textContent = encourageQuotes[Math.floor(Math.random()*encourageQuotes.length)];

// Đếm ký tự trong textarea
const inputEl = document.getElementById('inputText');
const charCountEl = document.getElementById('charCount');
function updateCharCount(){
  charCountEl.textContent = inputEl.value.length + '/1000';
}
inputEl.addEventListener('input', updateCharCount);
updateCharCount();

async function analyze(){
  const text = document.getElementById('inputText').value.trim();
  const errorEl = document.getElementById('errorMsg');
  errorEl.style.display = 'none';

  if(!text){
    errorEl.textContent = 'Vui lòng nhập nội dung.';
    errorEl.style.display = 'block';
    return;
  }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true; btn.innerHTML = '🧠 Đang lắng nghe...';

  try {
    const res = await fetch(API_BASE + '/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
    const data = await res.json();

    if(!res.ok){
      errorEl.textContent = data.error || 'Có lỗi xảy ra, vui lòng thử lại.';
      errorEl.style.display = 'block';
      btn.disabled = false; btn.innerHTML = '💛 Chia sẻ với mình <span class="arrow">→</span>';
      return;
    }

    lastResult = data;
    renderResult(data, text);
    history.unshift({text: text.slice(0,50)+(text.length>50?'...':''), level: data.prediction});
    renderHistory();

  } catch(err){
    errorEl.textContent = 'Không kết nối được tới máy chủ. Hãy chắc chắn Backend (app.py) đang chạy tại http://127.0.0.1:5000';
    errorEl.style.display = 'block';
  }

  btn.disabled = false; btn.innerHTML = '💛 Chia sẻ với mình <span class="arrow">→</span>';
}

function renderResult(data, originalText){
  document.getElementById('badge').className = 'badge ' + data.prediction;
  document.getElementById('badge').textContent = data.prediction;
  document.getElementById('confVal').textContent = (data.confidence*100).toFixed(1) + '%';
  document.getElementById('descText').textContent = data.result_description;

  document.getElementById('recommendation').innerHTML =
    '<ul class="rec-list">' + data.recommendations.map(r=>`<li>${r}</li>`).join('') + '</ul>';

  const order = ['Minimum','Mild','Moderate','Severe'];
  const barsEl = document.getElementById('probBars');
  barsEl.innerHTML = '';
  order.forEach(lv=>{
    const p = data.probabilities[lv] || 0;
    barsEl.innerHTML += `<div class="prob-row">
      <div class="prob-label">${lv}</div>
      <div class="prob-track"><div class="prob-fill ${lv}" id="fill-${lv}"><span>${(p*100).toFixed(0)}%</span></div></div>
    </div>`;
  });
  order.forEach(lv=>{
    const p = data.probabilities[lv] || 0;
    setTimeout(()=>{
      const el = document.getElementById('fill-'+lv);
      if(el){ el.style.width = (p*100).toFixed(0)+'%'; el.classList.add('filled'); }
    }, 100);
  });

  const kwEl = document.getElementById('ruleKeywords');
  if(data.highlight_words && data.highlight_words.length){
    kwEl.innerHTML = data.highlight_words.map(w=>`<span class="kw-chip">${w}</span>`).join('');
  } else {
    kwEl.innerHTML = '<span style="color:var(--ink-soft); font-size:13px;">Không tìm thấy từ khóa nào trong danh sách.</span>';
  }
  document.getElementById('limeResult').innerHTML = '';

  const crisisBox = document.getElementById('crisisBox');
  if(data.crisis_resources){
    crisisBox.style.display = 'block';
    document.getElementById('crisisList').innerHTML = data.crisis_resources.map(r=>{
      let actionBtn = '';
      if(r.phone) actionBtn += `<a class="crisis-btn" href="tel:${r.phone}">📞 Gọi ngay ${r.phone}</a>`;
      if(r.email) actionBtn += `<a class="crisis-btn secondary" href="mailto:${r.email}">✉️ Gửi email</a>`;
      if(r.messenger) actionBtn += `<a class="crisis-btn secondary" href="${r.messenger}" target="_blank">💬 Nhắn tin</a>`;
      return `<div class="crisis-item"><b>${r.name}</b><div class="crisis-actions">${actionBtn}</div><i>${r.note}</i></div>`;
    }).join('');
  } else {
    crisisBox.style.display = 'none';
  }

  document.getElementById('resultCard').classList.add('show');
  document.getElementById('resultCard').scrollIntoView({behavior:'smooth', block:'start'});
}

async function explainLime(){
  const text = document.getElementById('inputText').value.trim();
  if(!text) return;
  const btn = document.getElementById('explainBtn');
  const loading = document.getElementById('explainLoading');
  btn.disabled = true; loading.style.display = 'block';

  try {
    const res = await fetch(API_BASE + '/explain', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
    const data = await res.json();

    if(res.ok){
      const html = '<div class="method-note">Các từ này có ảnh hưởng thực sự đến dự đoán của mô hình (phương pháp LIME).</div>' +
        data.explanation.map(e=>`<span class="lime-chip">${e.word} (${e.weight>0?'+':''}${e.weight})</span>`).join('');
      document.getElementById('limeResult').innerHTML = html;
    } else {
      document.getElementById('limeResult').innerHTML = '<span style="color:var(--red);">'+(data.error||'Lỗi khi giải thích')+'</span>';
    }
  } catch(err){
    document.getElementById('limeResult').innerHTML = '<span style="color:var(--red);">Không kết nối được tới máy chủ.</span>';
  }

  btn.disabled = false; loading.style.display = 'none';
}

function renderHistory(){
  const el = document.getElementById('historyList');
  if(history.length===0) return;
  const colorMap = {Minimum:'var(--blue-deep)', Mild:'var(--sage-deep)', Moderate:'var(--rose-deep)', Severe:'var(--red)'};
  el.innerHTML = history.slice(0,5).map(h=>`
    <div class="history-item">
      <span class="history-text">${h.text}</span>
      <span class="history-badge" style="background:${colorMap[h.level]}">${h.level}</span>
    </div>`).join('');
}

function clearResult(){
  document.getElementById('resultCard').classList.remove('show');
  document.getElementById('inputText').value='';
  document.getElementById('errorMsg').style.display='none';
  updateCharCount();
}

function exportResult(){
  if(!lastResult) return;
  const text = document.getElementById('inputText').value;
  const recText = lastResult.recommendations.join(' | ');
  const content = `KẾT QUẢ PHÂN TÍCH\n------------------\nNội dung: ${text}\nMức độ: ${lastResult.prediction}\nĐộ tin cậy: ${(lastResult.confidence*100).toFixed(1)}%\nKhuyến nghị: ${recText}\n\n(Công cụ nghiên cứu học thuật, không thay thế chẩn đoán y khoa)`;
  const blob = new Blob([content], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ket-qua-phan-tich.txt';
  a.click();
}

// Nav link active state khi cuộn trang
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', () => {
  let current = 'hero';
  document.querySelectorAll('main section[id]').forEach(sec => {
    if (window.scrollY >= sec.offsetTop - 120) current = sec.id;
  });
  navLinks.forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === '#' + current);
  });
});