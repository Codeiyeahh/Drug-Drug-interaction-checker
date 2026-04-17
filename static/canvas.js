// Shared Neural Network Canvas Animation
(function () {
  const canvas = document.getElementById('canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const mouse = { x: null, y: null };

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  class Node {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * 0.5;
      this.vy = (Math.random() - 0.5) * 0.5;
      this.radius = Math.random() * 3 + 1.5;
      this.baseColor = Math.random() > 0.7 ? '#4ec3c9' : '#888';
    }
    update() {
      if (this.x < 0 || this.x > canvas.width) this.vx = -this.vx;
      if (this.y < 0 || this.y > canvas.height) this.vy = -this.vy;
      this.x += this.vx;
      this.y += this.vy;

      if (mouse.x !== null) {
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 200) {
          const angle = Math.atan2(dy, dx);
          const force = (200 - dist) / 200;
          this.vx -= Math.cos(angle) * force * 1.2;
          this.vy -= Math.sin(angle) * force * 1.2;
        }
      }

      // Speed cap
      const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      if (speed > 2) { this.vx = (this.vx / speed) * 2; this.vy = (this.vy / speed) * 2; }
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.baseColor;
      ctx.shadowBlur = 10;
      ctx.shadowColor = this.baseColor;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  const nodes = Array.from({ length: 100 }, () => new Node());

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    nodes.forEach((nodeA, i) => {
      nodes.slice(i + 1).forEach(nodeB => {
        const dx = nodeA.x - nodeB.x;
        const dy = nodeA.y - nodeB.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 220) {
          const opacity = (1 - distance / 220) * 0.6;
          const gradient = ctx.createLinearGradient(nodeA.x, nodeA.y, nodeB.x, nodeB.y);
          gradient.addColorStop(0, `rgba(78,195,201,${opacity})`);
          gradient.addColorStop(1, `rgba(136,136,136,${opacity})`);
          ctx.beginPath();
          ctx.moveTo(nodeA.x, nodeA.y);
          ctx.lineTo(nodeB.x, nodeB.y);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      });
    });

    nodes.forEach(node => { node.update(); node.draw(); });
    requestAnimationFrame(animate);
  }

  animate();

  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });
  window.addEventListener('mouseleave', () => { mouse.x = null; mouse.y = null; });
})();
