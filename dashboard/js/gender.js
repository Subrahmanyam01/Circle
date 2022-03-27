const genderData = {
  labels: ["Female", "Male"],
  datasets: [
    {
      data: [67, 33],
      backgroundColor: ["rgba(255, 99, 132, 0.2)", "rgba(75, 192, 192, 0.2)"],
      borderColor: ["rgb(255, 99, 132)", "rgb(75, 192, 192)"],

      hoverOffset: 4,
    },
  ],
};

const genderConfig = {
  type: "pie",
  data: genderData,
  options: {
    plugins: {
      legend: {
        // display: false,
        position: "bottom",
      },
    },
    tooltips: {
      enabled: false,
    },
  },
};

const genderChart = new Chart(document.getElementById("genderChart"), genderConfig);
