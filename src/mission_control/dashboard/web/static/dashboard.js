/* Mission Control -- Chart.js dashboard charts */

const CHART_COLORS = {
	accent: "#4fc3f7",
	success: "#66bb6a",
	error: "#ef5350",
	warning: "#ffa726",
	grid: "rgba(255,255,255,0.08)",
	text: "#888",
};

const COMMON_OPTIONS = {
	responsive: true,
	maintainAspectRatio: false,
	animation: { duration: 300 },
	plugins: {
		legend: {
			labels: { color: CHART_COLORS.text, font: { size: 11 } },
		},
	},
	scales: {
		x: {
			ticks: { color: CHART_COLORS.text, font: { size: 10 } },
			grid: { color: CHART_COLORS.grid },
		},
		y: {
			ticks: { color: CHART_COLORS.text, font: { size: 10 } },
			grid: { color: CHART_COLORS.grid },
			beginAtZero: true,
		},
	},
};

let chartScore = null;
let chartTests = null;
let chartCost = null;

function initCharts() {
	const scoreCtx = document.getElementById("chart-score");
	const testsCtx = document.getElementById("chart-tests");
	const costCtx = document.getElementById("chart-cost");

	if (!scoreCtx || !testsCtx || !costCtx) return;

	chartScore = new Chart(scoreCtx, {
		type: "line",
		data: {
			labels: [],
			datasets: [{
				label: "Score",
				data: [],
				borderColor: CHART_COLORS.accent,
				backgroundColor: "rgba(79,195,247,0.15)",
				fill: true,
				tension: 0.3,
				pointRadius: 3,
			}],
		},
		options: COMMON_OPTIONS,
	});

	chartTests = new Chart(testsCtx, {
		type: "bar",
		data: {
			labels: [],
			datasets: [
				{
					label: "Passed",
					data: [],
					backgroundColor: CHART_COLORS.success,
				},
				{
					label: "Failed",
					data: [],
					backgroundColor: CHART_COLORS.error,
				},
			],
		},
		options: {
			...COMMON_OPTIONS,
			scales: {
				...COMMON_OPTIONS.scales,
				x: { ...COMMON_OPTIONS.scales.x, stacked: true },
				y: { ...COMMON_OPTIONS.scales.y, stacked: true },
			},
		},
	});

	chartCost = new Chart(costCtx, {
		type: "bar",
		data: {
			labels: [],
			datasets: [{
				label: "Cost ($)",
				data: [],
				backgroundColor: CHART_COLORS.warning,
			}],
		},
		options: COMMON_OPTIONS,
	});
}

async function updateCharts() {
	try {
		const [scoreRes, testsRes, costRes] = await Promise.all([
			fetch("/api/score-history"),
			fetch("/api/test-trend"),
			fetch("/api/cost-per-round"),
		]);

		const scoreData = await scoreRes.json();
		const testsData = await testsRes.json();
		const costData = await costRes.json();

		if (chartScore) {
			chartScore.data.labels = scoreData.labels;
			chartScore.data.datasets[0].data = scoreData.data;
			chartScore.update("none");
		}

		if (chartTests) {
			chartTests.data.labels = testsData.labels;
			chartTests.data.datasets[0].data = testsData.passed;
			chartTests.data.datasets[1].data = testsData.failed;
			chartTests.update("none");
		}

		if (chartCost) {
			chartCost.data.labels = costData.labels;
			chartCost.data.datasets[0].data = costData.data;
			chartCost.update("none");
		}
	} catch (err) {
		console.error("Chart update failed:", err);
	}
}

document.addEventListener("DOMContentLoaded", function () {
	initCharts();
	updateCharts();
	setInterval(updateCharts, 5000);
});
