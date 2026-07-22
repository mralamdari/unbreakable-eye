/**
 * Cloudflare Worker for Telegram Bot API proxy.
 *
 * Forwards alerts from your local app to Telegram API.
 * Runs 24/7 on Cloudflare's edge network (free tier: 100K requests/day).
 *
 * Endpoints:
 *   POST /send     - Send text message (with optional inline buttons)
 *   POST /photo    - Send photo with caption
 *   GET  /health   - Health check
 *
 * Environment variables (set via wrangler secret):
 *   TELEGRAM_BOT_TOKEN - Your bot token from @BotFather
 *   ALLOWED_TOKENS     - Comma-separated list of valid tokens (security)
 *   DEFAULT_CHAT_ID    - Default chat ID to send messages to
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Health check
    if (request.method === 'GET' && url.pathname === '/health') {
      return new Response(JSON.stringify({ status: 'ok', service: 'unbreakable-eye-telegram' }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Only accept POST for send/photo
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    try {
      const body = await request.json();
      const { message, token, photo, chat_id, buttons } = body;

      // Validate token
      const allowedTokens = (env.ALLOWED_TOKENS || '').split(',').map(t => t.trim());
      if (!token || !allowedTokens.includes(token)) {
        return new Response('Unauthorized', { status: 401 });
      }

      const botToken = env.TELEGRAM_BOT_TOKEN;
      const targetChatId = chat_id || env.DEFAULT_CHAT_ID;

      if (!botToken || !targetChatId) {
        return new Response('Server configuration error', { status: 500 });
      }

      // Send photo with caption
      if (photo) {
        const result = await sendTelegramPhoto(botToken, targetChatId, photo, message || '', buttons);
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Send text message
      if (message) {
        const result = await sendTelegramMessage(botToken, targetChatId, message, buttons);
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json' }
        });
      }

      return new Response('No message or photo provided', { status: 400 });

    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
};

async function sendTelegramMessage(botToken, chatId, text, buttons) {
  const url = `https://api.telegram.org/bot${botToken}/sendMessage`;

  const payload = {
    chat_id: chatId,
    text: text,
    parse_mode: 'Markdown'
  };

  // Add inline keyboard if buttons provided
  if (buttons && Array.isArray(buttons) && buttons.length > 0) {
    payload.reply_markup = {
      inline_keyboard: buttons
    };
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  return await response.json();
}

async function sendTelegramPhoto(botToken, chatId, photoUrl, caption, buttons) {
  const url = `https://api.telegram.org/bot${botToken}/sendPhoto`;

  const payload = {
    chat_id: chatId,
    photo: photoUrl,
    caption: caption,
    parse_mode: 'Markdown'
  };

  // Add inline keyboard if buttons provided
  if (buttons && Array.isArray(buttons) && buttons.length > 0) {
    payload.reply_markup = {
      inline_keyboard: buttons
    };
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  return await response.json();
}
