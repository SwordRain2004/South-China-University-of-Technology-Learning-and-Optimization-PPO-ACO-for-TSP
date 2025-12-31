import subprocess
import time
import os
import re
import pandas as pd

# ===== 1. 定义实验配置 =====
experiments = [
    {"name": "baseline", "args": []},
    {"name": "fix_alpha", "args": ["--fix_alpha"]},
    {"name": "fix_beta", "args": ["--fix_beta"]},
    {"name": "fix_rho", "args": ["--fix_rho"]},
]

def run_experiments(exp_name, args, mode="train"):
    print(f"{'=' * 20} Experiment  Info {'=' * 20}")
    print(f"Experiment: {exp_name} | Mode: {mode}")
    print(f"Parameters: {' '.join(args) if args else 'Default'}")

    # 构建命令
    cmd = ["python", "main.py", "--mode", mode] + args

    # 添加时间戳以便记录
    start_time = time.time()

    try:
        # 运行命令
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 强制使用 utf-8 解码
            errors='replace'  # 遇到无法解码的字符（如进度条乱码）用 ? 替换，不报错
        )

        # 计算运行时间
        elapsed_time = time.time() - start_time

        # 保存结果到日志文件
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{exp_name}_{mode}.log")
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Parameters: {' '.join(args) if args else 'None'}\n")
            f.write(f"Duration: {elapsed_time:.2f}s\n")
            f.write("=" * 50 + "\n")
            f.write("Standard Output:\n")
            f.write(result.stdout + "\n")
            f.write("=" * 50 + "\n")
            f.write("Standard Error:\n")
            f.write(result.stderr + "\n")

        print(f"✓ Completed! Duration: {elapsed_time:.2f}s")
        print(f"  Log saved to: {log_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment Failed! Exit Code: {e.returncode}")
        print(f"  Error Message: {e.stderr}")
        return False


def read_text_auto(path):
    for enc in ("utf-8", "gbk", "gb2312"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode file: {path}")


def experiments_result_combine():
    print(f"\n{'=' * 60}")
    print("Aggregating experiment results...")

    txt_dir = "./results/"
    if not os.path.exists(txt_dir):
        print(f"Error: Result directory '{txt_dir}' does not exist. Please run experiments first.")
        return

    output_excel = "ppo_aco_solution_quality_summary.xlsx"
    records = []

    pattern = {
        "Mean": re.compile(r"Mean\s*=\s*([-\d.]+)"),
        "Min": re.compile(r"Min\s*=\s*([-\d.]+)"),
        "Max": re.compile(r"Max\s*=\s*([-\d.]+)"),
        "Std": re.compile(r"Std\s*=\s*([-\d.]+)")
    }

    for fname in os.listdir(txt_dir):
        if not fname.endswith("_solution_quality_comparison.txt"):
            continue

        file_path = os.path.join(txt_dir, fname)

        content = read_text_auto(file_path)

        record = {
            "Experiment": fname.replace("_solution_quality_comparison.txt", "")
        }

        for key, regex in pattern.items():
            match = regex.search(content)
            record[key] = float(match.group(1)) if match else None

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(by="Mean", ascending=False)
    df.to_excel(os.path.join(txt_dir, output_excel), index=False)

    print(f"Saved as CSV instead: {os.path.join(txt_dir, output_excel)}")


def main():
    """主流程：运行所有消融实验并汇总"""
    print("Starting PPO-ACO Ablation Study Automation Script")
    print(f"Total planned experiments: {len(experiments)}")
    print("Note: Ensure parameters in Config.py are set reasonably (small epochs recommended for quick verification)")

    success_count = 0

    for i, exp in enumerate(experiments):
        exp_name = exp["name"]
        args = exp["args"]

        print(f"\n>>> Progress: {i + 1}/{len(experiments)} - {exp_name}")

        if run_experiments(exp_name, args, "train"):
            if run_experiments(exp_name, args, "test"):
                success_count += 1
            else:
                print(f"⚠ Warning: Experiment '{exp_name}' Testing Stage Failed")
        else:
            print(f"⚠ Warning: Experiment '{exp_name}' Training Stage Failed, Skipping Test")

    # 生成实验总结
    print(f"\n{'=' * 60}")
    print(f"All Batches Execution Completed")
    print(f"{'=' * 60}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed/Skipped: {len(experiments) - success_count}")

    # 最后汇总结果
    experiments_result_combine()


if __name__ == "__main__":
    main()
