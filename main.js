// Any photo whose file doesn't exist yet turns into a labelled placeholder,
// so you can see exactly which filename to drop into /images.
function markEmpty(img) {
  const fig = img.closest(".shot");
  if (!fig) return;
  const path = img.getAttribute("src");
  fig.classList.add("empty");
  fig.innerHTML = `<span>+ photo</span><span>${path}</span>`;
}

document.querySelectorAll(".shot img").forEach((img) => {
  if (img.complete && img.naturalWidth === 0) markEmpty(img);
  else img.addEventListener("error", () => markEmpty(img), { once: true });
});

// Hairline under the nav once you scroll.
const nav = document.querySelector(".nav");
const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 8);
onScroll();
window.addEventListener("scroll", onScroll, { passive: true });

// Gentle fade-in for sections.
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        io.unobserve(e.target);
      }
    });
  },
  { rootMargin: "0px 0px -8% 0px" }
);
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

document.querySelectorAll("[data-year]").forEach((el) => {
  el.textContent = new Date().getFullYear();
});

// Hero video: only play while it's on screen, and not at all if the visitor prefers reduced motion.
const robot = document.querySelector("video.robot");
if (robot) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    robot.removeAttribute("autoplay");
    robot.pause();
  } else {
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting) robot.play().catch(() => {});
      else robot.pause();
    }).observe(robot);
  }
}
