from pathlib import Path
import sys

import config
from src.utils import connect_duckdb
from src.pipeline import FeasibilityPipeline


def main():
    print('=' * 72)
    print('MIMIC-IV severe CAP / corticosteroid feasibility - data check')
    print('=' * 72)
    print(f'DATA_ROOT = {config.DATA_ROOT}')
    if not Path(config.DATA_ROOT).exists():
        print('\n[ERROR] 数据根目录不存在。请修改 config.py 中的 DATA_ROOT。')
        sys.exit(1)

    con = connect_duckdb(config.PROJECT_ROOT, config.DUCKDB_THREADS)
    pipe = FeasibilityPipeline(con, config)
    try:
        pipe.discover_tables()
        df = pipe.inventory_dataframe()
        print('\n找到的数据表：')
        print(df[['table', 'size_gb', 'status']].to_string(index=False))
        out = config.OUTPUT_DIR / '00_data_inventory.csv'
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'\n[OK] 核心表齐全。清单已保存：{out}')
        print('下一步运行：01_run_feasibility.py')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        raise
    finally:
        con.close()


if __name__ == '__main__':
    main()
