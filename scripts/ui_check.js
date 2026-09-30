// Drives the real UI in Edge against the real gateway. Usage: node scripts/ui_check.js
const path = require("path"), fs = require("fs");
const core = process.env.PLAYWRIGHT_CORE || require.resolve("playwright-core");
const { chromium } = require(core);
const key = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "quanta.config.json"), "utf8")).server.apiKey;
const out = path.join(__dirname, "..", "workspace");
(async () => {
  const browser = await chromium.launch({ executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = []; page.on("pageerror", e => errors.push(String(e))); page.on("console", m => m.type() === "error" && errors.push(m.text()));
  const ok = (n, c, d = "") => console.log(`[${c ? "PASS" : "FAIL"}] ${n} ${d}`);
  await page.goto("http://127.0.0.1:8788/");
  await page.fill("#key-input", key); await page.click("#key-form button[type=submit]");
  await page.waitForFunction(() => document.getElementById("conn-pill").dataset.state === "ok", null, { timeout: 15000 }).catch(() => {});
  ok("connects with key", (await page.getAttribute("#conn-pill", "data-state")) === "ok", await page.textContent("#conn-text"));
  ok("provider cards from API", (await page.locator(".provider-card").count()) === 5);
  ok("freebuff marked unavailable", (await page.locator(".provider-card", { hasText: "freebuff" }).textContent()).includes("No headless mode"));
  await page.screenshot({ path: path.join(out, "ui-top.png") });
  // playground: plain streaming message on opencode
  await page.selectOption("#pg-provider", "opencode"); await page.fill("#pg-model", "default");
  await page.fill("#pg-input", "Reply with exactly: pong"); await page.click("#pg-send");
  await page.waitForSelector(".msg.assistant .bubble", { timeout: 120000 });
  await page.waitForFunction(() => !document.getElementById("pg-send").disabled, null, { timeout: 120000 });
  ok("streamed reply", /pong/i.test(await page.locator(".msg.assistant .bubble").last().textContent()), await page.locator(".msg.assistant .badge").first().textContent());
  // switch provider mid-conversation and recall context
  await page.fill("#pg-input", "Remember the codeword KIWI-7. Reply OK."); await page.click("#pg-send");
  await page.waitForFunction(() => !document.getElementById("pg-send").disabled && document.querySelectorAll(".msg.assistant").length >= 2, null, { timeout: 120000 });
  await page.selectOption("#pg-provider", "cline"); await page.fill("#pg-model", "default");
  await page.fill("#pg-input", "What was the codeword? Answer with the codeword only."); await page.click("#pg-send");
  await page.waitForFunction(() => !document.getElementById("pg-send").disabled && document.querySelectorAll(".msg.assistant").length >= 3, null, { timeout: 120000 });
  const badges = await page.locator(".msg.assistant .badge:first-child").allTextContents();
  ok("provider switch keeps context", /kiwi-7/i.test(await page.locator(".msg.assistant .bubble").last().textContent()), badges.join(" | "));
  // tools
  await page.click("#pg-clear"); await page.check("#pg-tools"); await page.uncheck("#pg-stream");
  await page.selectOption("#pg-provider", "opencode");
  await page.fill("#pg-input", "What is the weather in Paris? Use the tool."); await page.click("#pg-send");
  await page.waitForSelector(".toolcall form", { timeout: 120000 });
  ok("tool call card shown", (await page.locator(".toolcall pre").first().textContent()).toLowerCase().includes("paris"));
  await page.fill(".toolcall input", '{"temp_c":23}'); await page.click(".toolcall form button");
  await page.waitForFunction(() => document.querySelectorAll(".msg.assistant .bubble").length >= 1 && !document.getElementById("pg-send").disabled, null, { timeout: 120000 });
  ok("tool result continues conversation", /23/.test(await page.locator(".msg.assistant .bubble").last().textContent()));
  await page.waitForTimeout(500);
  await page.evaluate(() => document.getElementById("activity").scrollIntoView());
  await page.waitForTimeout(7000);
  ok("activity table populated", (await page.locator("#activity-rows tr").count()) >= 3);
  // bad key -> error state
  await page.fill("#key-input", "definitely-not-the-key-000000"); await page.click("#key-form button[type=submit]");
  await page.waitForFunction(() => document.getElementById("conn-pill").dataset.state === "bad", null, { timeout: 15000 }).catch(() => {});
  ok("bad key shows rejected", (await page.textContent("#conn-text")).includes("rejected"));
  await page.fill("#key-input", key); await page.click("#key-form button[type=submit]");
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(out, "ui-full.png"), fullPage: true });
  await page.setViewportSize({ width: 400, height: 800 }); await page.waitForTimeout(300);
  ok("no horizontal overflow at 400px", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.screenshot({ path: path.join(out, "ui-mobile.png") });
  ok("no console/page errors", errors.filter(e => !/401|Unauthorized|Failed to load resource/.test(e)).length === 0, errors.join(" ; ").slice(0, 200));
  await browser.close();
})().catch(e => { console.error("SCRIPT ERROR", e.message); process.exit(1); });
