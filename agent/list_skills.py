"""List skills without importing RoboTwin, SAPIEN, CuRobo, or CUDA."""

from robot_skills import RobotSkills


def main() -> None:
    print("当前可用基础技能：")
    for signature, description in RobotSkills.describe().items():
        print(f"  - {signature}: {description}")


if __name__ == "__main__":
    main()
