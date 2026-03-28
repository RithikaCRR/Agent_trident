/**
 * AgentTrader AI — Frontend Application Logic
 * Handles API communication, chart rendering, and UI updates.
 */

const API = "/api";

// ─── State ────────────────────────────────────────────────────────────────────
let isRunning = false;
let pollInterval = null;
let priceChart = null;

// ─── DOM Elements ─────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);

const elBalanceValue     = $("balance-value");
const elBalanceChange    = $("balance-change");
const elActionBadge      = $("action-badge");
const elStockName        = $("stock-name");
const elConfidenceBar    = $("confidence-bar");
const elDecisionConf     = $("decision-confidence");
const elBtnStart         = $("btn-start");
const elBtnStop          = $("btn-stop");
const elCycleInfo        = $("cycle-info");
const elStatusPill       = $("status-pill");
const elClock            = $("live-clock");
const elSummarySection   = $("summary-section");
const elDecisionSummary  = $("decision-summary");
const elDissentingViews  = $("dissenting-views");
const elChartSymbol      = $("chart-symbol");
const elErrorBanner      = $("error-banner");
const elErrorText        = $("error-text");

// Agent elements
const elTechBadge    = $("tech-badge");
const elTechReasoning = $("tech-reasoning");
const elTechSignals  = $("tech-signals");
const elSentBadge    = $("sent-badge");
const elSentReasoning = $("sent-reasoning");
const elSentSignals  = $("sent-signals");
const elRiskBadge    = $("risk-badge");
const elRiskReasoning = $("risk-reasoning");
const elRiskSignals  = $("risk-signals");


// ─── Live Clock ───────────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    elClock.textContent = now.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
    });
}
setInterval(updateClock, 1000);
updateClock();


// ─── Chart Setup ──────────────────────────────────────────────────────────────
function initChart() {
    const ctx = document.getElementById("price-chart").getContext("2d");

    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
    gradient.addColorStop(0, "rgba(52, 211, 153, 0.25)");
    gradient.addColorStop(1, "rgba(52, 211, 153, 0.0)");

    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Close Price",
                data: [],
                borderColor: "#34d399",
                backgroundColor: gradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: "#34d399",
                fill: true,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: "index",
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "#94a3b8",
                    bodyColor: "#f1f5f9",
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `$${ctx.parsed.y.toFixed(2)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.03)" },
                    ticks: {
                        color: "#64748b",
                        font: { size: 10 },
                        maxRotation: 45,
                        maxTicksLimit: 12,
                    },
                },
                y: {
                    grid: { color: "rgba(255,255,255,0.03)" },
                    ticks: {
                        color: "#64748b",
                        font: { size: 10 },
                        callback: (v) => "$" + v.toFixed(0),
                    },
                },
            },
        },
    });
}


// ─── Update Chart ─────────────────────────────────────────────────────────────
function updateChart(prices) {
    if (!priceChart || !prices || prices.length === 0) return;

    const labels = prices.map((p) => {
        const t = p.time;
        // Show short date labels
        if (t.length > 10) return t.slice(11, 16);
        return t.slice(5);
    });

    const data = prices.map((p) => p.close);

    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = data;
    priceChart.update("none");
}


// ─── Action Badge Color ───────────────────────────────────────────────────────
function setActionBadge(el, action) {
    const act = (action || "HOLD").toUpperCase();
    el.textContent = act;
    el.className = el.className.replace(/\b(buy|sell|hold)\b/g, "");
    el.classList.add(act.toLowerCase());
}


// ─── Render Signals ───────────────────────────────────────────────────────────
function renderSignals(container, signals) {
    container.innerHTML = "";
    if (!signals || !Array.isArray(signals)) return;
    signals.forEach((s) => {
        const tag = document.createElement("span");
        tag.className = "signal-tag";
        tag.textContent = s;
        container.appendChild(tag);
    });
}


// ─── Format Currency ──────────────────────────────────────────────────────────
function formatCurrency(val) {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
    }).format(val);
}


// ─── Render Portfolio ─────────────────────────────────────────────────────────
function renderPortfolio(portfolio) {
    const container = document.getElementById("portfolio-container");
    if (!container) return;
    
    if (!portfolio || Object.keys(portfolio).length === 0) {
        container.innerHTML = '<div class="log-empty">No stocks currently owned.</div>';
        return;
    }

    container.innerHTML = "";
    Object.keys(portfolio).forEach((symbol) => {
        const data = portfolio[symbol];
        if (data.shares <= 0) return;

        const item = document.createElement("div");
        item.className = "portfolio-item";

        const symEl = document.createElement("div");
        symEl.className = "portfolio-symbol";
        symEl.textContent = symbol;

        const sharesEl = document.createElement("div");
        sharesEl.className = "portfolio-shares";
        sharesEl.textContent = `${data.shares.toFixed(4)} shares @ $${data.avg_price.toFixed(2)}`;

        const valEl = document.createElement("div");
        valEl.className = "portfolio-value";
        // Estimate value based on average price 
        const estValue = data.shares * data.avg_price;
        valEl.textContent = `Est. Input: ${formatCurrency(estValue)}`;

        item.appendChild(symEl);
        item.appendChild(sharesEl);
        item.appendChild(valEl);
        container.appendChild(item);
    });

    if (container.children.length === 0) {
        container.innerHTML = '<div class="log-empty">No stocks currently owned.</div>';
    }
}


// ─── Render Transaction History ───────────────────────────────────────────────
function renderHistory(trades) {
    const container = document.getElementById("transaction-history");
    if (!container) return;
    
    if (!trades || trades.length === 0) {
        container.innerHTML = '<div class="log-empty">No transactions yet...</div>';
        return;
    }

    container.innerHTML = "";
    // Show newest first
    const reversed = [...trades].reverse();
    
    reversed.forEach((trade) => {
        const row = document.createElement("div");
        row.className = "history-entry";

        const time = document.createElement("div");
        time.className = "hist-time";
        time.textContent = trade.time.length > 10 ? trade.time.slice(11, 19) : trade.time;

        const sym = document.createElement("div");
        sym.className = "hist-symbol";
        sym.textContent = trade.symbol;

        const action = document.createElement("div");
        action.className = `hist-action ${trade.action.toLowerCase()}`;
        action.textContent = trade.action;

        const price = document.createElement("div");
        price.className = "hist-price";
        price.textContent = `$${trade.price.toFixed(2)}`;

        const bal = document.createElement("div");
        bal.className = "hist-balance";
        bal.textContent = `$${trade.balance_after.toFixed(2)}`;

        row.appendChild(time);
        row.appendChild(sym);
        row.appendChild(action);
        row.appendChild(price);
        row.appendChild(bal);
        container.appendChild(row);
    });
}


// ─── Fetch Status ─────────────────────────────────────────────────────────────
async function fetchStatus() {
    try {
        const resp = await fetch(`${API}/status`);
        const data = await resp.json();

        // Balance
        const balance = data.balance || 10000;
        elBalanceValue.textContent = formatCurrency(balance);
        const diff = balance - 10000;
        if (diff !== 0) {
            const sign = diff > 0 ? "+" : "";
            elBalanceChange.textContent = `${sign}${formatCurrency(diff)} from initial`;
            elBalanceChange.style.color = diff > 0 ? "var(--green)" : "var(--red)";
        }

        // Decision
        if (data.decision) {
            setActionBadge(elActionBadge, data.decision.action);
            elDecisionConf.textContent = `Confidence: ${data.decision.confidence}%`;
            elConfidenceBar.style.width = `${data.decision.confidence}%`;

            if (data.decision.summary) {
                elSummarySection.style.display = "block";
                elDecisionSummary.textContent = data.decision.summary;
                elDissentingViews.textContent = data.decision.dissenting_views || "";
            }
        }

        // Stock name
        if (data.symbol) {
            elStockName.textContent = data.symbol;
            elChartSymbol.textContent = data.symbol;
        }

        // Agents
        if (data.agents) {
            updateAgentCard(data.agents.technical, elTechBadge, elTechReasoning, elTechSignals, "Technical Analyst");
            updateAgentCard(data.agents.sentiment, elSentBadge, elSentReasoning, elSentSignals, "Sentiment Analyst");
            updateAgentCard(data.agents.risk, elRiskBadge, elRiskReasoning, elRiskSignals, "Risk Manager");
        }

        // Cycle info
        const lastUp = data.last_update
            ? new Date(data.last_update).toLocaleTimeString()
            : "—";
        elCycleInfo.textContent = `Cycle: ${data.cycle_count} | Last update: ${lastUp}`;

        // Error
        if (data.error) {
            elErrorBanner.style.display = "flex";
            elErrorText.textContent = data.error;
        } else {
            elErrorBanner.style.display = "none";
        }

        // Running status
        isRunning = data.running;
        updateControlsUI();
        
        // Ensure polling is active if running
        if (isRunning && !pollInterval) {
            startPolling();
        } else if (!isRunning && pollInterval) {
            stopPolling();
        }

        // Portfolio
        if (data.portfolio) {
            renderPortfolio(data.portfolio);
        }

        // Agent Logs
        if (data.agent_logs) {
            renderAgentLogs(data.agent_logs);
        }

    } catch (err) {
        console.error("Status fetch error:", err);
    }
}


// ─── Fetch History (Chart & Trades) ───────────────────────────────────────────
async function fetchHistory() {
    try {
        const resp = await fetch(`${API}/history`);
        const data = await resp.json();
        if (data.prices) {
            updateChart(data.prices);
        }
        if (data.trades) {
            renderHistory(data.trades);
        }
    } catch (err) {
        console.error("History fetch error:", err);
    }
}


// ─── Update Agent Card ────────────────────────────────────────────────────────
function updateAgentCard(agent, badgeEl, reasoningEl, signalsEl, agentName) {
    if (!agent) return;
    setActionBadge(badgeEl, agent.action);
    const reasoningText = agent.reasoning || "No reasoning provided.";
    reasoningEl.innerHTML = `<strong>${agentName}:</strong> ${reasoningText}`;
    renderSignals(signalsEl, agent.key_signals);
}


// ─── Render Agent Logs ────────────────────────────────────────────────────────
function renderAgentLogs(logs) {
    const container = document.getElementById("agent-log-container");
    if (!logs || logs.length === 0) return;

    container.innerHTML = "";
    logs.forEach((entry) => {
        const row = document.createElement("div");
        row.className = "log-entry";

        const time = document.createElement("span");
        time.className = "log-time";
        time.textContent = entry.time;

        const agent = document.createElement("span");
        agent.className = "log-agent";
        const agentLower = (entry.agent || "").toLowerCase();
        if (agentLower.includes("technical")) agent.classList.add("technical");
        else if (agentLower.includes("sentiment")) agent.classList.add("sentiment");
        else if (agentLower.includes("risk")) agent.classList.add("risk");
        else if (agentLower.includes("moderator")) agent.classList.add("moderator");
        else agent.classList.add("system");
        agent.textContent = entry.agent;

        const msg = document.createElement("span");
        msg.className = "log-message";
        msg.textContent = entry.message;

        row.appendChild(time);
        row.appendChild(agent);
        row.appendChild(msg);
        container.appendChild(row);
    });

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}


// ─── Controls UI ──────────────────────────────────────────────────────────────
function updateControlsUI() {
    elBtnStart.disabled = isRunning;
    elBtnStop.disabled = !isRunning;

    if (isRunning) {
        elStatusPill.classList.add("active");
        elStatusPill.querySelector(".status-text").textContent = "Live";
    } else {
        elStatusPill.classList.remove("active");
        elStatusPill.querySelector(".status-text").textContent = "Offline";
    }
}


// ─── Start / Stop ─────────────────────────────────────────────────────────────
elBtnStart.addEventListener("click", async () => {
    elBtnStart.disabled = true;
    try {
        await fetch(`${API}/start`, { method: "POST" });
        isRunning = true;
        updateControlsUI();
        startPolling();
        // Fetch initial status after a short delay
        setTimeout(() => {
            fetchStatus();
            fetchHistory();
        }, 2000);
    } catch (err) {
        console.error("Start error:", err);
        elBtnStart.disabled = false;
    }
});

elBtnStop.addEventListener("click", async () => {
    elBtnStop.disabled = true;
    try {
        await fetch(`${API}/stop`, { method: "POST" });
        isRunning = false;
        updateControlsUI();
        stopPolling();
    } catch (err) {
        console.error("Stop error:", err);
        elBtnStop.disabled = false;
    }
});


// ─── Polling ──────────────────────────────────────────────────────────────────
function startPolling() {
    stopPolling();
    // Poll status & chart every 15 seconds for snappy updates;
    // backend still re-analyzes every 2 minutes
    pollInterval = setInterval(() => {
        fetchStatus();
        fetchHistory();
    }, 15000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}


// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    fetchStatus();
    fetchHistory();
});
