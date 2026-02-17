import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Draw blocks
def draw_block(x, y, width, height, text):
    rect = Rectangle((x, y), width, height, fill=True, color="#add8e6", ec="black")
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, weight='bold')

draw_block(1, 2, 2, 1, "Controller")
draw_block(4.5, 2, 2, 1, "Robot Dynamics")
draw_block(8, 2, 1.5, 1, "Output")

draw_block(4.5, 4, 2, 1, "Sensors")

# Draw arrows
def draw_arrow(x_start, y_start, x_end, y_end):
    arrow = FancyArrow(x_start, y_start, x_end-x_start, y_end-y_start, width=0.05, length_includes_head=True, color="black")
    ax.add_patch(arrow)

# Forward path
draw_arrow(3, 2.5, 4.5, 2.5)  # Controller -> Dynamics
draw_arrow(6.5, 2.5, 8, 2.5)  # Dynamics -> Output

# Feedback path
draw_arrow(8.75, 2.5, 8.75, 4.5)  # Output -> Sensors
draw_arrow(6.5, 4.5, 3, 3.5)      # Sensors -> Controller
draw_arrow(0.5, 2.5, 1, 2.5)      # Setpoint -> Controller
ax.text(0, 2.5, "Setpoint", ha='right', va='center', fontsize=12)

plt.title("Closed-Loop Robot Control Visualization", fontsize=14, weight='bold')
plt.show()
