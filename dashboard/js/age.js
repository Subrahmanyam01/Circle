const ageData = {
  labels: ["2-10", "10-25", "25-45", "45-60", ">60"],
  datasets: [
    {
      data: [65, 59, 80, 81, 56],
      backgroundColor: [
        "rgba(255, 99, 132, 0.2)",
        "rgba(255, 159, 64, 0.2)",
        "rgba(255, 205, 86, 0.2)",
        "rgba(75, 192, 192, 0.2)",
        "rgba(54, 162, 235, 0.2)",
        "rgba(153, 102, 255, 0.2)",
        "rgba(201, 203, 207, 0.2)",
      ],
      borderColor: [
        "rgb(255, 99, 132)",
        "rgb(255, 159, 64)",
        "rgb(255, 205, 86)",
        "rgb(75, 192, 192)",
        "rgb(54, 162, 235)",
        "rgb(153, 102, 255)",
        "rgb(201, 203, 207)",
      ],
      borderWidth: 1,
    },
  ],
};

const ageConfig = {
  type: "bar",
  data: ageData,
  options: {
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
      gridLines: {
        display: false,
        borderWidth: 3,
      },
    },
  },
};

const ageChart = new Chart(document.getElementById("ageChart"), ageConfig);
