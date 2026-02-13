/* Mission Control -- Chart.js dashboard charts (multi-project) */

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

// Per-project chart instances
const projectCharts = {};

function initProjectCharts(projectName) {
	// Destroy existing charts for this project (in case of tab switch)
	if (projectCharts[projectName]) {
		Object.values(projectCharts[projectName]).forEach(c => c.destroy());
	}

	const scoreCtx = document.getElementById(`chart-score-${projectName}`);
	const testsCtx = document.getElementById(`chart-tests-${projectName}`);
	const costCtx = document.getElementById(`chart-cost-${projectName}`);

	if (!scoreCtx || !testsCtx || !costCtx) return;

	projectCharts[projectName] = {
		score: new Chart(scoreCtx, {
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
		}),
		tests: new Chart(testsCtx, {
			type: "bar",
			data: {
				labels: [],
				datasets: [
					{ label: "Passed", data: [], backgroundColor: CHART_COLORS.success },
					{ label: "Failed", data: [], backgroundColor: CHART_COLORS.error },
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
		}),
		cost: new Chart(costCtx, {
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
		}),
	};

	// Initial data fetch
	updateProjectCharts(projectName);
}

async function updateProjectCharts(projectName) {
	const charts = projectCharts[projectName];
	if (!charts) return;

	try {
		const base = `/project/${projectName}/api`;
		const [scoreRes, testsRes, costRes] = await Promise.all([
			fetch(`${base}/score-history`),
			fetch(`${base}/test-trend`),
			fetch(`${base}/cost-per-round`),
		]);

		const scoreData = await scoreRes.json();
		const testsData = await testsRes.json();
		const costData = await costRes.json();

		if (charts.score) {
			charts.score.data.labels = scoreData.labels;
			charts.score.data.datasets[0].data = scoreData.data;
			charts.score.update("none");
		}

		if (charts.tests) {
			charts.tests.data.labels = testsData.labels;
			charts.tests.data.datasets[0].data = testsData.passed;
			charts.tests.data.datasets[1].data = testsData.failed;
			charts.tests.update("none");
		}

		if (charts.cost) {
			charts.cost.data.labels = costData.labels;
			charts.cost.data.datasets[0].data = costData.data;
			charts.cost.update("none");
		}
	} catch (err) {
		console.error(`Chart update failed for ${projectName}:`, err);
	}
}

// Legacy support: init charts for single-project view
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
				{ label: "Passed", data: [], backgroundColor: CHART_COLORS.success },
				{ label: "Failed", data: [], backgroundColor: CHART_COLORS.error },
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

// Periodic chart updates for active project tabs
setInterval(function() {
	// Update all project charts that are currently visible
	for (const name of Object.keys(projectCharts)) {
		const el = document.querySelector(`[data-project="${name}"]`);
		if (el) {
			updateProjectCharts(name);
		}
	}
	// Legacy charts
	if (chartScore) updateCharts();
}, 5000);

document.addEventListener("DOMContentLoaded", function () {
	initCharts();
	updateCharts();
});
