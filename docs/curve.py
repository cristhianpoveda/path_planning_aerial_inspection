import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured constants, Table "Measured optical constants" in main.tex
b0, kdof, df, kr, fx = 1.461, 2.768, 1.567, 1.234, 1431.85
wt = 0.004                      # 4 mm defect, per the mechanism section
dmin, dmax, dcc = 1.02, 2.25, 1.78

def Q(d, defocus=True):
    c = kdof * np.abs(d - df) / d if defocus else 0.0
    btot = np.sqrt(b0**2 + c**2)
    return wt / (kr * btot * d / fx)

d = np.linspace(dmin, 3.05, 4000)
q = Q(d)
qnd = Q(d, defocus=False)

ipk = np.argmax(q)
dpk, qpk = d[ipk], q[ipk]

# five-node Gauss-Hermite, sigma_d = 0.20 m, exaggerated for legibility
xi = np.array([-2.856970, -1.355626, 0.0, 1.355626, 2.856970])
w5 = np.array([0.011257, 0.222076, 0.533333, 0.222076, 0.011257])
sd = 0.13
dn = dpk + xi * sd
qn = Q(dn)
EQ = np.sum(w5 * qn)

fig, ax = plt.subplots(figsize=(7.2, 4.3))

ax.plot(d, q, lw=2.6, color="#1D9E75", label="with defocus", zorder=5)
ax.plot(d, qnd, lw=1.8, color="#888780", ls="--",
        label="resolution only", zorder=4)

ax.axvspan(dcc, 3.05, color="#888780", alpha=0.10, lw=0)
ax.axvline(dcc, color="#888780", lw=1.0, ls=":", zorder=2)
ax.text(dcc + 0.04, ax.get_ylim()[0], "", va="bottom")

# spread of delivered depths
inner = (dn > dpk - 2.2 * sd) & (dn < dpk + 2.2 * sd)
ax.plot(dn[inner], qn[inner], "o", ms=5.5, mfc="white",
        mec="#888780", mew=1.3, zorder=6)
ax.plot([dn[1], dn[3]], [qn[1], qn[3]], color="#D85A30", lw=1.2,
        ls="--", zorder=6)

ax.plot([dpk], [qpk], "o", ms=9, color="#1D9E75", zorder=8)
ax.plot([dpk], [EQ], "o", ms=9, color="#D85A30", zorder=8)

ax.annotate("planned", xy=(dpk, qpk), xytext=(dpk - 0.20, qpk + 0.42),
            color="#1D9E75", fontsize=11,
            arrowprops=dict(arrowstyle="-", color="#1D9E75", lw=0.9))
ax.annotate("delivered", xy=(dpk, EQ), xytext=(dpk + 0.30, EQ - 0.55),
            color="#D85A30", fontsize=11,
            arrowprops=dict(arrowstyle="-", color="#D85A30", lw=0.9))

ax.text(dcc + 0.06, qpk * 1.05, "concavity limit\n1.78 m",
        fontsize=10, color="#5F5E5A", va="top")
ax.text(dpk, 0.12, "peak\n1.23 m", fontsize=10, color="#5F5E5A",
        ha="center", va="bottom")

ax.set_xlabel("standoff  $d$  (m)", fontsize=12)
ax.set_ylabel("$w_t / w_{min}$", fontsize=12)
ax.set_xlim(dmin, 3.05)
ax.set_ylim(0, qpk * 1.32)
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#B4B2A9")
ax.tick_params(colors="#5F5E5A", labelsize=10)

fig.tight_layout()
fig.savefig("quality_curve.pdf")
fig.savefig("quality_curve.png", dpi=220)

print("peak d = %.3f   Q = %.3f" % (dpk, qpk))
print("sigma_d = %.2f   E[Q] = %.4f   Q(E[d]) = %.4f   gap = %.2f%%"
      % (sd, EQ, qpk, 100 * (1 - EQ / qpk)))