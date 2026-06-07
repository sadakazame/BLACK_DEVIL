import subprocess
import sys
import shutil
import os

def run_analysis(file_path):
    """静的解析（Pyflakes, Mypy）を実行"""
    tools = [
        {"name": "Pyflakes", "cmd": [sys.executable, "-m", "pyflakes", file_path]},
        {"name": "Mypy", "cmd": [sys.executable, "-m", "mypy", file_path]},
    ]
    for tool in tools:
        res = subprocess.run(tool["cmd"], capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            print(f"❌ {file_path} - {tool['name']} エラー:\n{res.stderr}")
            return False
    print(f"✅ {file_path}: 静的解析クリア")
    return True

def run_integration_test(main_file, expected_output):
    """実際に動かして、2つのファイルの連携が正しいかチェック"""
    print(f"[動作検証中: {main_file}]")
    try:
        result = subprocess.run([sys.executable, main_file], capture_output=True, text=True, encoding="utf-8")
        output = result.stdout.strip()
        if expected_output in output:
            print(f"✅ 連携テスト成功: 出力に '{expected_output}' を確認しました")
            return True
        else:
            print(f"❌ 連携テスト失敗: 期待値 '{expected_output}' が見つかりません。出力: {output}")
            return False
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def deploy_all(files, dest_dir):
    """全て合格したら指定のフォルダへ一括コピー"""
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dest_dir, f))
    print(f"🚀 全ての検証に合格！ {len(files)}個のファイルを {dest_dir} へ配置しました。")

if __name__ == "__main__":
    target_files = ["logic.py", "main_app.py"]
    dist_dir = "./production_dist"
    
    # 1. 各ファイルの静的解析
    all_ok = True
    for f in target_files:
        if not run_analysis(f):
            all_ok = False
            
    # 2. 2つを合わせた時の動作検証（10 * 5 = 50 になるか）
    if all_ok:
        if not run_integration_test("main_app.py", "RESULT:50"):
            all_ok = False
            
    # 3. 最終配置
    print("-" * 40)
    if all_ok:
        deploy_all(target_files, dist_dir)
    else:
        print("🚫 検証に失敗したため、配置を中止しました。")