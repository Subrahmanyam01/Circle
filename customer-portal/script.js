var count = 1;

document.getElementById("add-cart").addEventListener("click", () => {
  document.getElementById("num").innerHTML = `Add to cart &nbsp; (${count++})`;
});
