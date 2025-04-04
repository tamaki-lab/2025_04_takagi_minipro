# miniproject-template

研究室で使用するプロジェクトのテンプレートです．

## 使い方

1. このリポジトリをフォークする．
2. フォークしたリポジトリをクローンする．
3. リポジトリの名前を変更する．
4. ファイルを編集して，自分のプロジェクトに合わせた内容に変更する．
5. 仮想環境を作成し，依存パッケージをインストールする．
6. プロジェクトを開始する．

## ディレクトリ構成

```text
.
├── .comet.config
├── .flake8
├── .gitignore
├── .mypy.ini
├── .pep8
├── .pylintrc
├── .pytest.ini
├── .vscode
│   ├── extensions.json
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
├── README.md
├── data
├── pyproject.toml
├── requirements.txt
└── src
    └── main.py
```

### ファイル構成

- `.comet.config`: comet.mlを使用する際の設定ファイル．
- `.flake8`: コードスタイルチェッカーflake8の設定ファイル．
- `.gitignore`: gitで管理しないファイルやディレクトリを指定するファイル．
- `.mypy.ini`: 静的型チェッカーmypyの設定ファイル．
- `.pep8`: コードフォーマッタautopepの設定ファイル．
- `.pylintrc`: 静的コード解析ツールpylintの設定ファイル．
- `.pytest.ini`: テストフレームワークpytestの設定ファイル．
- `.vscode`: VSCodeの設定が保存されているディレクトリ．
  - `extensions.json`: VSCodeのおすすめの拡張機能．
  - `launch.json`: VSCodeのデバッグの設定．
  - `settings.json`: VSCode自体の設定．
  - `tasks.json`: VSCodeのタスクの設定．
- `README.md`: プロジェクトの説明や使い方などが記述されたREADME．
- `data`: プロジェクトで使用するデータを格納するディレクトリ．
- `pyproject.toml`: プロジェクトの依存関係やビルド設定などが記述されたファイル．コードチェッカーやフォーマッターの設定もここに記述されている場合がある．
- `requirements.txt`: プロジェクトに必要なPythonパッケージが記述されたファイル．`pip install -r requirements.txt`でインストールする．
- `src`: プロジェクトのソースコードを格納するディレクトリ．
  - `main.py`: プロジェクトのmainファイル．

### 仮想環境の準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 仮想環境を有効化する
source .venv/bin/activate
```

```bash
# 仮想環境を無効化する
deactivate
```
