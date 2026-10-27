
from city_env import CityFleetEnv

env = CityFleetEnv()

state, info = env.reset(seed=42)

print("Initial state:", state)
print("Initial information:", info)
print()
print(env.render())

# Try a few random actions.
for step in range(10):
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(
        f"Step {step + 1}: "
        f"Action={action}, "
        f"Reward={reward}, "
        f"Position={info['vehicle_position']}, "
        f"Trips={info['completed_trips']}"
    )

    print(env.render())
    print()

env.close()
