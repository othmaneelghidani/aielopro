require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const crypto = require('crypto');

const app = express();
app.use(express.json());
app.use(cors());
app.use(express.static(__dirname));

const {
  GITHUB_TOKEN,
  GITHUB_OWNER,
  GITHUB_REPO,
  GEMINI_API_KEY,
  ADMIN_SECRET
} = process.env;

// إعداد الاتصال بـ GitHub
const githubApi = axios.create({
  baseURL: `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents`,
  headers: {
    Authorization: `Bearer ${GITHUB_TOKEN}`,
    Accept: 'application/vnd.github.v3+json'
  }
});

let CACHE = {
  memory: { items: [] },
  keys: { users: {} },
  config: {
    persona: 'أنت مساعد ذكي، دقيق، ومباشر. تعتمد على الحقائق الموثقة من بحث جوجل وتجيب بأسلوب واضح ومنظم.',
    similarityThreshold: 0.84
  },
  shas: {},
  activeModel: 'gemini-3.8-flash' // الموديل الرسمي الجديد
};

async function loadOrInitGithubFile(filename, defaultData) {
  try {
    const res = await githubApi.get(`/${filename}`);
    const content = Buffer.from(res.data.content, 'base64').toString('utf8');
    CACHE.shas[filename] = res.data.sha;
    return JSON.parse(content);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      console.log(`[GitHub] جاري إنشاء ملف ${filename} لأول مرة...`);
      await syncToGithub(filename, defaultData, `تهيئة ملف ${filename}`);
      return defaultData;
    }
    console.error(`[خطأ GitHub] فشل قراءة ${filename}:`, err.message);
    return defaultData;
  }
}

async function syncToGithub(filename, data, commitMessage) {
  try {
    const contentBase64 = Buffer.from(JSON.stringify(data, null, 2), 'utf8').toString('base64');
    const payload = {
      message: commitMessage,
      content: contentBase64
    };
    if (CACHE.shas[filename]) {
      payload.sha = CACHE.shas[filename];
    }
    const res = await githubApi.put(`/${filename}`, payload);
    CACHE.shas[filename] = res.data.content.sha;
    console.log(`[✅ GitHub Sync] تم حفظ ${filename}: ${commitMessage}`);
  } catch (err) {
    if (err.response && err.response.status === 409) {
      const latest = await githubApi.get(`/${filename}`);
      CACHE.shas[filename] = latest.data.sha;
      return syncToGithub(filename, data, commitMessage);
    }
    console.error(`[❌ GitHub Error] فشل حفظ ${filename}:`, err.response?.data?.message || err.message);
  }
}

async function detectBestGeminiModel() {
  const cleanKey = (GEMINI_API_KEY || '').trim();
  try {
    const res = await axios.get(`https://generativelanguage.googleapis.com/v1beta/models?key=${cleanKey}`, {
      headers: { 'x-goog-api-key': cleanKey }
    });
    const models = res.data.models || [];
    
    const supported = models
      .filter(m => m.supportedGenerationMethods && m.supportedGenerationMethods.includes('generateContent'))
      .map(m => m.name.replace('models/', ''));

    console.log('🤖 الموديلات المتاحة فحسابك:', supported.slice(0, 6).join(', '));

    // الترتيب الجديد على حساب تحديثات جوجل 2026
    const preferred = ['gemini-3.8-flash', 'gemini-3.8-pro', 'gemini-3.5-flash', 'gemini-3.0-flash'];
    for (const p of preferred) {
      const found = supported.find(s => s === p || s.startsWith(p));
      if (found) {
        CACHE.activeModel = found;
        console.log(`✅ تم اختيار الموديل الشغال تلقائياً: ${CACHE.activeModel}`);
        return;
      }
    }

    if (supported.length > 0) {
      CACHE.activeModel = supported[0];
      console.log(`✅ تم اختيار الموديل الأول المتاح: ${CACHE.activeModel}`);
    }
  } catch (err) {
    console.log(`ℹ️ سيتم استخدام الموديل الافتراضي الحديث: ${CACHE.activeModel}`);
  }
}

async function initBrain() {
  console.log('🔄 جاري ربط الاتصال مع GitHub وتحميل الذاكرة...');
  await detectBestGeminiModel();
  CACHE.memory = await loadOrInitGithubFile('memory.json', { items: [] });
  CACHE.keys = await loadOrInitGithubFile('api_keys.json', { users: {} });
  CACHE.config = await loadOrInitGithubFile('config.json', CACHE.config);
  console.log(`🧠 الذاكرة واجدة! عدد الأسئلة المتعلمة: ${CACHE.memory.items.length}`);
}

// 1. تحويل السؤال لأرقام (Vector Embedding)
async function getEmbedding(text) {
  const cleanKey = (GEMINI_API_KEY || '').trim();
  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${cleanKey}`;
    const res = await axios.post(url, {
      model: 'models/text-embedding-004',
      content: { parts: [{ text: text.trim() }] }
    }, {
      headers: { 'x-goog-api-key': cleanKey, 'Content-Type': 'application/json' }
    });
    return res.data.embedding.values;
  } catch (err) {
    return null;
  }
}

function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeText(text) {
  return text.trim().toLowerCase()
    .replace(/[؟?!.,،؛"']/g, '')
    .replace(/[أإآ]/g, 'ا')
    .replace(/ة/g, 'ه')
    .replace(/\s+/g, ' ');
}

// بحث مجاني سريع في الويب
async function freeWebSearch(query) {
  try {
    const wikiUrl = `https://ar.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&utf8=&format=json`;
    const res = await axios.get(wikiUrl, { timeout: 3500 });
    const snippets = (res.data.query?.search || []).slice(0, 4).map(s => s.snippet.replace(/<\/?[^>]+(>|$)/g, ""));
    return snippets.length > 0 ? snippets.join('\n') : '';
  } catch {
    return '';
  }
}

// 4. التوليد والبحث الذكي باستخدام gemini-3.8-flash
async function searchAndAnswerWithGemini(question, customInstruction = '') {
  const cleanKey = (GEMINI_API_KEY || '').trim();
  
  // قائمة الموديلات الجديدة مرتبة بالأولوية
  const modelsToTry = [...new Set([
    CACHE.activeModel || 'gemini-3.8-flash',
    'gemini-3.8-flash',
    'gemini-3.8-pro',
    'gemini-3.5-flash'
  ])];

  const webContext = await freeWebSearch(question);
  const systemPrompt = `${CACHE.config.persona}\n${customInstruction ? `توجيه إضافي: ${customInstruction}` : ''}\n${webContext ? `معلومات مساعدة من البحث:\n${webContext}\n` : ''}أجب على السؤال التالي بدقة ووضوح:`;

  let lastError = null;

  for (const modelName of modelsToTry) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${cleanKey}`;
    const headers = { 'x-goog-api-key': cleanKey, 'Content-Type': 'application/json' };

    // المحاولة 1: مع أداة بحث جوجل المباشر (Google Search Grounding)
    try {
      const res = await axios.post(url, {
        contents: [{ parts: [{ text: `${systemPrompt}\n\nالسؤال: ${question}` }] }],
        tools: [{ google_search: {} }]
      }, { headers });

      const candidate = res.data.candidates[0];
      const answerText = candidate.content.parts.map(p => p.text).join('\n');
      const chunks = candidate.groundingMetadata?.groundingChunks || [];
      const sources = chunks.filter(c => c.web?.uri).map(c => ({ title: c.web.title, url: c.web.uri }));

      return { answer: answerText, sources };
    } catch (err1) {
      // المحاولة 2: بدون أداة google_search فحالة ما إذا كان الموديل يطلب استدعاء عادي
      try {
        const res2 = await axios.post(url, {
          contents: [{ parts: [{ text: `${systemPrompt}\n\nالسؤال: ${question}` }] }]
        }, { headers });

        return {
          answer: res2.data.candidates[0].content.parts[0].text,
          sources: []
        };
      } catch (err2) {
        lastError = err2.response?.data?.error?.message || err2.message;
        console.warn(`⚠️ الموديل ${modelName} رجع خطأ:`, lastError);
      }
    }
  }

  throw new Error(`خطأ من Google API: ${lastError}`);
}

// ================= حماية الـ API =================
async function verifyApiKey(req, res, next) {
  const apiKey = req.headers['x-api-key'] || req.query.api_key || req.body.api_key;

  if (apiKey === 'PUBLIC_WEB_CHAT') {
    req.clientUser = 'زائر الموقع';
    return next();
  }

  if (!apiKey) {
    return res.status(401).json({ error: 'مفقود API Key! للحصول على مفتاح تواصل عبر إنستغرام: othmane_elghidani' });
  }

  const keyData = CACHE.keys.users[apiKey];
  if (!keyData || !keyData.active) {
    return res.status(403).json({ error: 'هاد الـ API Key غير صالح أو تم توقيفه من طرف الإدارة.' });
  }

  const today = new Date().toISOString().split('T')[0];
  if (keyData.lastDate !== today) {
    keyData.lastDate = today;
    keyData.todayCount = 0;
  }

  if (keyData.dailyLimit > 0 && keyData.todayCount >= keyData.dailyLimit) {
    return res.status(429).json({ error: `وصلتي للحد الأقصى اليومي ديالك (${keyData.dailyLimit} سؤال فاليوم).` });
  }

  keyData.todayCount += 1;
  keyData.totalUsage = (keyData.totalUsage || 0) + 1;
  keyData.lastUsed = new Date().toISOString();

  syncToGithub('api_keys.json', CACHE.keys, `استخدام المفتاح: ${keyData.owner}`);

  req.clientUser = keyData.owner;
  next();
}

function verifyAdmin(req, res, next) {
  const secret = req.headers['x-admin-secret'] || req.body.adminSecret;
  if (secret !== ADMIN_SECRET) {
    return res.status(403).json({ error: 'كلمة سر الأدمن غالطة!' });
  }
  next();
}

// ================= مسار الذكاء الاصطناعي الرئيسي (/api/ask) =================
app.post('/api/ask', verifyApiKey, async (req, res) => {
  const { question, forceRefresh = false, instruction = '' } = req.body;
  if (!question || question.trim() === '') {
    return res.status(400).json({ error: 'المرجو كتابة السؤال فـ question' });
  }

  try {
    const startTime = Date.now();
    const normQ = normalizeText(question);

    if (!forceRefresh) {
      const exactMatch = CACHE.memory.items.find(item => item.normalized === normQ);
      if (exactMatch) {
        exactMatch.hits = (exactMatch.hits || 1) + 1;
        return res.json({
          status: 'success',
          source: '⚡ ذاكرة GitHub المباشرة (أوتوماتيكي 100%)',
          similarity: '100%',
          responseTime: `${Date.now() - startTime}ms`,
          question: question,
          matchedQuestion: exactMatch.question,
          answer: exactMatch.answer,
          sources: exactMatch.sources || []
        });
      }
    }

    const queryVector = await getEmbedding(question);
    if (!forceRefresh && queryVector && CACHE.memory.items.length > 0) {
      let bestMatch = null;
      let highestScore = 0;

      for (const item of CACHE.memory.items) {
        if (item.embedding) {
          const score = cosineSimilarity(queryVector, item.embedding);
          if (score > highestScore) {
            highestScore = score;
            bestMatch = item;
          }
        }
      }

      if (bestMatch && highestScore >= CACHE.config.similarityThreshold) {
        bestMatch.hits = (bestMatch.hits || 1) + 1;
        return res.json({
          status: 'success',
          source: '🧠 ذاكرة GitHub الذكية (تطابق فالمعنى)',
          similarity: `${(highestScore * 100).toFixed(1)}%`,
          responseTime: `${Date.now() - startTime}ms`,
          question: question,
          matchedQuestion: bestMatch.question,
          answer: bestMatch.answer,
          sources: bestMatch.sources || []
        });
      }
    }

    console.log(`🔍 [سؤال جديد] البحث والتوليد لـ: "${question}"`);
    const { answer, sources } = await searchAndAnswerWithGemini(question, instruction);

    const newMemoryItem = {
      id: crypto.randomBytes(6).toString('hex'),
      question: question.trim(),
      normalized: normQ,
      answer: answer,
      sources: sources,
      embedding: queryVector,
      askedBy: req.clientUser,
      hits: 1,
      createdAt: new Date().toISOString()
    };

    const existingIndex = CACHE.memory.items.findIndex(i => i.normalized === normQ);
    if (existingIndex !== -1) {
      CACHE.memory.items[existingIndex] = newMemoryItem;
    } else {
      CACHE.memory.items.unshift(newMemoryItem);
    }

    syncToGithub('memory.json', CACHE.memory, `تعلم سؤال جديد: ${question.substring(0, 40)}`);

    res.json({
      status: 'success',
      source: '🌐 بحث جوجل الحي + تم الحفظ فـ GitHub',
      similarity: 'سؤال جديد (تم تعلمه)',
      responseTime: `${Date.now() - startTime}ms`,
      question: question,
      answer: answer,
      sources: sources
    });

  } catch (err) {
    console.error('❌ تفاصيل الخطأ:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ================= مسارات لوحة التحكم =================
app.get('/api/admin/dashboard', verifyAdmin, (req, res) => {
  const memoryPreview = CACHE.memory.items.map(item => ({
    id: item.id,
    question: item.question,
    answer: item.answer,
    askedBy: item.askedBy,
    hits: item.hits || 1,
    createdAt: item.createdAt
  }));

  res.json({
    config: CACHE.config,
    keys: CACHE.keys.users,
    memoryCount: CACHE.memory.items.length,
    memory: memoryPreview
  });
});

app.post('/api/admin/keys/create', verifyAdmin, async (req, res) => {
  const { owner, dailyLimit = 100 } = req.body;
  const newKey = 'ai_free_' + crypto.randomBytes(12).toString('hex');

  CACHE.keys.users[newKey] = {
    owner: owner || 'مستخدم جديد',
    dailyLimit: Number(dailyLimit),
    todayCount: 0,
    totalUsage: 0,
    active: true,
    createdAt: new Date().toISOString()
  };

  await syncToGithub('api_keys.json', CACHE.keys, `إنشاء مفتاح جديد لـ ${owner}`);
  res.json({ apiKey: newKey, data: CACHE.keys.users[newKey] });
});

app.post('/api/admin/keys/toggle', verifyAdmin, async (req, res) => {
  const { apiKey, action } = req.body;
  if (!CACHE.keys.users[apiKey]) return res.status(404).json({ error: 'الساروت ما كاينش' });

  if (action === 'delete') {
    delete CACHE.keys.users[apiKey];
  } else {
    CACHE.keys.users[apiKey].active = !CACHE.keys.users[apiKey].active;
  }

  await syncToGithub('api_keys.json', CACHE.keys, `تعديل حالة المفتاح ${apiKey}`);
  res.json({ success: true, keys: CACHE.keys.users });
});

app.post('/api/admin/config', verifyAdmin, async (req, res) => {
  const { persona, similarityThreshold } = req.body;
  if (persona) CACHE.config.persona = persona;
  if (similarityThreshold) CACHE.config.similarityThreshold = Number(similarityThreshold);

  await syncToGithub('config.json', CACHE.config, 'تحديث إعدادات وتوجيه الذكاء الاصطناعي');
  res.json({ success: true, config: CACHE.config });
});

app.post('/api/admin/memory/delete', verifyAdmin, async (req, res) => {
  const { id } = req.body;
  CACHE.memory.items = CACHE.memory.items.filter(item => item.id !== id);
  await syncToGithub('memory.json', CACHE.memory, `حذف سؤال من الذاكرة: ${id}`);
  res.json({ success: true });
});

const PORT = process.env.PORT || 3000;
initBrain().then(() => {
  app.listen(PORT, () => {
    console.log(`🚀 السيرفر خدام! حل المتصفح فـ: http://localhost:${PORT}`);
  });
});