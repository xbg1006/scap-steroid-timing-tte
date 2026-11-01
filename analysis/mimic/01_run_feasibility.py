from pathlib import Path
import traceback

import config
from src.utils import connect_duckdb
from src.pipeline import FeasibilityPipeline


def main():
    print('=' * 78)
    print('Severe CAP + early corticosteroid feasibility pipeline')
    print('=' * 78)
    print(f'Data root : {config.DATA_ROOT}')
    print(f'Cache dir : {config.CACHE_DIR}')
    print(f'Output dir: {config.OUTPUT_DIR}')
    print('\n本脚本会：')
    print('  1) 建立成人早期 ICU CAP proxy 队列')
    print('  2) 从 chartevents 提取呼吸支持（大型表只缓存研究相关 itemid）')
    print('  3) 从 inputevents 提取升压药与 ICU 激素记录')
    print('  4) 从 prescriptions 提取系统激素医嘱作为敏感性来源')
    print('  5) 以首次器官支持为 severe_t0，统计 24h 内激素暴露')
    print('  6) 输出 respiratory-only / shock-only / combined 三类 phenotype')

    con = connect_duckdb(config.PROJECT_ROOT, config.DUCKDB_THREADS)
    pipe = FeasibilityPipeline(con, config)
    try:
        print('\n[1/6] 检查数据文件...')
        pipe.discover_tables()
        inventory = pipe.inventory_dataframe()
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(config.OUTPUT_DIR / '00_data_inventory.csv', index=False, encoding='utf-8-sig')

        print('[2/6] 建立/读取呼吸事件缓存...')
        resp_cache = pipe.build_resp_cache()
        print(f'      {resp_cache}')

        print('[3/6] 建立/读取升压药/激素 inputevents 缓存...')
        input_cache = pipe.build_input_cache()
        print(f'      {input_cache}')

        print('[4/6] 建立/读取系统激素 prescriptions 缓存...')
        rx_cache = pipe.build_rx_steroid_cache()
        print(f'      {rx_cache}')

        print('[5/6] 构建 CAP、器官支持、激素暴露与 phenotype...')
        pipe.build_base_cohort()
        pipe.build_support_events(resp_cache, input_cache)
        pipe.build_steroid_exposure(input_cache, rx_cache)
        pipe.build_final_patient_table()

        print('[6/6] 导出结果...')
        paths = pipe.export_outputs(inventory)

        flow = pipe.make_flow_table()
        phen = pipe.make_phenotype_table()
        gonogo = pipe.make_go_nogo()
        print('\n=== Cohort flow ===')
        print(flow.to_string(index=False))
        print('\n=== Phenotypes ===')
        print(phen.to_string(index=False))
        print('\n=== GO / NO-GO ===')
        print(gonogo.to_string(index=False))
        print('\n[完成] 重点先看：')
        print(f"  {paths['excel']}")
        print(f"  {paths['gonogo_csv']}")
        print(f"  {paths['cohort_parquet']}")
        print('\n注意：这只是 feasibility，不是最终论文队列。正式分析前还要加强 CAP 影像/感染证据、既往激素适应证、实际给药、time-zero 与因果设计。')
    except Exception:
        print('\n[运行失败] 完整错误如下：\n')
        traceback.print_exc()
        raise
    finally:
        con.close()


if __name__ == '__main__':
    main()
