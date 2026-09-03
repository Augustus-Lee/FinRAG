"""导入示例数据字典（开发 / 演示 / 面试 Demo 用）。

用法:
    python -m finrag.scripts.import_dictionary

说明:
    - 幂等：表按 table_name 存在则更新，字段按 (table_id, field_name) 存在则更新；
    - 覆盖 3 张业务表（销售 / 客户 / 交易），共 100+ 字段，
      表/列白名单由 MCP Server 端管控；
    - synonyms 同时包含中文口径词与英文别名，供 SchemaLinker / 字典检索使用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag.db.init_db import init_db  # noqa: E402
from finrag.db.session import SessionLocal  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.models import DictField, DictTable  # noqa: E402

logger = get_logger("finrag.import_dictionary")

# ---------------------------------------------------------------------------
# 表元数据
# ---------------------------------------------------------------------------
TABLES = [
    {
        "table_name": "product_sales",
        "business_domain": "销售",
        "description": "产品销售事实表（日粒度）：覆盖线上线下全渠道销售明细汇总。",
        "owner": "零售事业部",
    },
    {
        "table_name": "customer_account",
        "business_domain": "客户",
        "description": "客户账户主数据：客户基本信息、资产、风险等级与 KYC 状态。",
        "owner": "财富管理部",
    },
    {
        "table_name": "transaction_record",
        "business_domain": "交易",
        "description": "证券交易流水表：委托、成交、清算全链路交易明细。",
        "owner": "运营结算部",
    },
]

# ---------------------------------------------------------------------------
# 字段元数据（synonyms 为检索同义词，is_sensitive 控制脱敏）
# ---------------------------------------------------------------------------
FIELDS: dict[str, list[dict]] = {
    "product_sales": [
        {"field_name": "trade_date", "field_type": "DATE", "comment": "交易/销售日期", "calibre": "自然日口径", "synonyms": ["交易日期", "销售日期", "成交日期", "日期"]},
        {"field_name": "product_code", "field_type": "VARCHAR(32)", "comment": "产品唯一编码", "synonyms": ["产品代码", "产品编号", "SKU"]},
        {"field_name": "product_name", "field_type": "VARCHAR(128)", "comment": "产品名称", "synonyms": ["产品名称", "品名", "商品名称"]},
        {"field_name": "category", "field_type": "VARCHAR(32)", "comment": "产品类别（如 手机/家电/美妆）", "synonyms": ["类别", "产品分类", "品类"]},
        {"field_name": "channel", "field_type": "VARCHAR(16)", "comment": "销售渠道：线上/线下/APP/门店", "synonyms": ["渠道", "销售渠道", "下单渠道"]},
        {"field_name": "region", "field_type": "VARCHAR(32)", "comment": "销售地区/省份", "synonyms": ["地区", "区域", "省份"]},
        {"field_name": "sales_amount", "field_type": "DECIMAL(18,2)", "comment": "含税销售总金额", "calibre": "含增值税；发生退货时冲减当期", "synonyms": ["销售金额", "销售额", "成交金额", "收入"]},
        {"field_name": "sales_quantity", "field_type": "INT", "comment": "销售数量（件/台）", "synonyms": ["数量", "销售数量", "件数", "销量"]},
        {"field_name": "cost_amount", "field_type": "DECIMAL(18,2)", "comment": "销售成本金额", "calibre": "按移动加权平均成本结转", "synonyms": ["成本", "销售成本", "成本金额"]},
        {"field_name": "gross_profit", "field_type": "DECIMAL(18,2)", "comment": "毛利金额 = 销售金额 - 成本金额", "calibre": "含税口径毛利，未扣费用", "synonyms": ["毛利", "毛利润", "毛利额"]},
        {"field_name": "discount_amount", "field_type": "DECIMAL(18,2)", "comment": "优惠/折扣金额", "synonyms": ["优惠", "折扣金额", "让利"]},
        {"field_name": "order_count", "field_type": "INT", "comment": "订单笔数", "synonyms": ["订单数量", "订单笔数", "单量"]},
        {"field_name": "customer_type", "field_type": "VARCHAR(16)", "comment": "客户类型：个人/企业", "synonyms": ["客户类型", "客群", "客户属性"]},
        {"field_name": "sales_rep", "field_type": "VARCHAR(32)", "comment": "销售代表/客户经理", "synonyms": ["销售员", "销售人员", "客户经理", "业务员"]},
        {"field_name": "warehouse_code", "field_type": "VARCHAR(32)", "comment": "发货仓库编码", "synonyms": ["仓库", "仓库代码", "仓编码"]},
        {"field_name": "logistics_company", "field_type": "VARCHAR(64)", "comment": "物流承运商", "synonyms": ["快递", "物流商", "承运商"]},
        {"field_name": "payment_method", "field_type": "VARCHAR(16)", "comment": "支付方式：微信/支付宝/银行卡/货到付款", "synonyms": ["支付渠道", "付款方式", "支付"]},
        {"field_name": "is_returned", "field_type": "TINYINT", "comment": "是否退货：1 是 / 0 否", "synonyms": ["退货标识", "退单标记", "是否退单"]},
        {"field_name": "is_gift", "field_type": "TINYINT", "comment": "是否赠品：1 是 / 0 否", "synonyms": ["赠品标记", "是否赠送"]},
        {"field_name": "promotion_name", "field_type": "VARCHAR(64)", "comment": "促销活动名称", "synonyms": ["活动", "营销活动", "促销"]},
        {"field_name": "unit_price", "field_type": "DECIMAL(12,2)", "comment": "成交单价", "synonyms": ["单价", "成交单价", "销售单价"]},
        {"field_name": "tax_rate", "field_type": "DECIMAL(5,2)", "comment": "增值税率（%）", "synonyms": ["税率", "增值税率"]},
        {"field_name": "batch_no", "field_type": "VARCHAR(32)", "comment": "生产批次号", "synonyms": ["批次", "生产批次", "批号"]},
        {"field_name": "expiry_date", "field_type": "DATE", "comment": "保质期到期日", "synonyms": ["到期日", "保质期", "过期日"]},
        {"field_name": "supplier_code", "field_type": "VARCHAR(32)", "comment": "供应商编码", "synonyms": ["供应商", "供应商代码"]},
        {"field_name": "store_code", "field_type": "VARCHAR(32)", "comment": "门店编码", "synonyms": ["门店", "门店号", "店铺编码"]},
        {"field_name": "settle_date", "field_type": "DATE", "comment": "结算/对账日期", "synonyms": ["对账日期", "结算日期"]},
        {"field_name": "invoice_no", "field_type": "VARCHAR(32)", "comment": "发票号码", "synonyms": ["发票号", "发票号码"]},
        {"field_name": "remark", "field_type": "VARCHAR(255)", "comment": "备注", "synonyms": ["备注", "说明"]},
        {"field_name": "sync_time", "field_type": "DATETIME", "comment": "数据同步时间（ETL）", "synonyms": ["同步时间", "更新时间"]},
        {"field_name": "etl_batch", "field_type": "VARCHAR(32)", "comment": "ETL 批次标识", "synonyms": ["批处理标识", "数据批次"]},
        {"field_name": "is_deleted", "field_type": "TINYINT", "comment": "逻辑删除标记：0 有效 / 1 删除", "synonyms": ["删除标记", "逻辑删除"]},
        {"field_name": "create_time", "field_type": "DATETIME", "comment": "记录创建时间", "synonyms": ["创建时间", "入库时间"]},
        {"field_name": "update_time", "field_type": "DATETIME", "comment": "记录更新时间", "synonyms": ["更新时间", "修改时间"]},
    ],
    "customer_account": [
        {"field_name": "customer_no", "field_type": "VARCHAR(32)", "comment": "客户号（唯一）", "synonyms": ["客户编号", "客户ID", "账号", "客户"]},
        {"field_name": "customer_name", "field_type": "VARCHAR(64)", "comment": "客户姓名", "synonyms": ["姓名", "客户名称", "用户名"]},
        {"field_name": "id_card", "field_type": "VARCHAR(18)", "comment": "身份证号", "is_sensitive": True, "synonyms": ["身份证", "证件号", "身份证号码"]},
        {"field_name": "mobile", "field_type": "VARCHAR(20)", "comment": "手机号", "is_sensitive": True, "synonyms": ["手机", "联系电话", "手机号码", "电话"]},
        {"field_name": "email", "field_type": "VARCHAR(128)", "comment": "电子邮箱", "is_sensitive": True, "synonyms": ["邮箱", "邮箱地址", "邮件"]},
        {"field_name": "gender", "field_type": "VARCHAR(8)", "comment": "性别：男/女", "synonyms": ["性别"]},
        {"field_name": "birthday", "field_type": "DATE", "comment": "出生日期", "synonyms": ["生日", "出生年月"]},
        {"field_name": "age", "field_type": "INT", "comment": "年龄", "synonyms": ["年龄", "岁数"]},
        {"field_name": "address", "field_type": "VARCHAR(255)", "comment": "联系地址", "is_sensitive": True, "synonyms": ["地址", "居住地址"]},
        {"field_name": "risk_level", "field_type": "VARCHAR(16)", "comment": "风险承受能力等级", "calibre": "按最新一次风险评估结果：保守型/稳健型/成长型/激进型", "synonyms": ["风险等级", "风险评级", "风险承受能力", "测评等级"]},
        {"field_name": "customer_level", "field_type": "VARCHAR(16)", "comment": "客户等级：普通/黄金/铂金/钻石", "synonyms": ["客户级别", "会员等级", "等级"]},
        {"field_name": "open_date", "field_type": "DATE", "comment": "开户日期", "synonyms": ["开户时间", "注册时间", "开户日"]},
        {"field_name": "open_channel", "field_type": "VARCHAR(16)", "comment": "开户渠道", "synonyms": ["开户渠道", "注册渠道"]},
        {"field_name": "branch_code", "field_type": "VARCHAR(32)", "comment": "营业部/分支机构编码", "synonyms": ["营业部", "分支机构", "网点"]},
        {"field_name": "broker_no", "field_type": "VARCHAR(32)", "comment": "客户经理/理财顾问编号", "synonyms": ["理财经理", "客户经理", "服务人员"]},
        {"field_name": "total_asset", "field_type": "DECIMAL(20,2)", "comment": "总资产", "calibre": "当前时点口径：证券市值 + 现金 + 理财持仓", "synonyms": ["资产总额", "总资产值", "资产"]},
        {"field_name": "available_asset", "field_type": "DECIMAL(20,2)", "comment": "可用资产/可用资金", "synonyms": ["可用资金", "可用余额", "可用"]},
        {"field_name": "frozen_asset", "field_type": "DECIMAL(20,2)", "comment": "冻结资产", "synonyms": ["冻结资金", "冻结金额"]},
        {"field_name": "total_liability", "field_type": "DECIMAL(20,2)", "comment": "总负债", "synonyms": ["负债总额", "负债"]},
        {"field_name": "net_asset", "field_type": "DECIMAL(20,2)", "comment": "净资产 = 总资产 - 总负债", "synonyms": ["净资产值", "净资产"]},
        {"field_name": "annual_income", "field_type": "DECIMAL(20,2)", "comment": "年收入水平", "is_sensitive": True, "synonyms": ["年化收入", "年收入", "收入水平"]},
        {"field_name": "occupation", "field_type": "VARCHAR(32)", "comment": "职业类型", "synonyms": ["职业", "行业", "职业类别"]},
        {"field_name": "education", "field_type": "VARCHAR(16)", "comment": "学历", "synonyms": ["学历", "文化程度"]},
        {"field_name": "marital_status", "field_type": "VARCHAR(8)", "comment": "婚姻状况", "synonyms": ["婚姻状态", "婚否"]},
        {"field_name": "kyc_status", "field_type": "VARCHAR(16)", "comment": "KYC/实名认证状态：已认证/未认证/待补充", "synonyms": ["实名认证状态", "认证状态", "KYC"]},
        {"field_name": "risk_assessment_date", "field_type": "DATE", "comment": "最近一次风险评估日期", "synonyms": ["测评日期", "风险评估日"]},
        {"field_name": "risk_assessment_expire", "field_type": "DATE", "comment": "风险评估到期日", "synonyms": ["测评到期日", "评估有效期"]},
        {"field_name": "auto_invest_flag", "field_type": "TINYINT", "comment": "自动理财签约标识：1 已签约 / 0 未签约", "synonyms": ["自动申购标记", "自动理财"]},
        {"field_name": "credit_score", "field_type": "INT", "comment": "信用评分", "is_sensitive": True, "synonyms": ["征信评分", "信用分"]},
        {"field_name": "blacklist_flag", "field_type": "TINYINT", "comment": "黑名单标记：1 命中 / 0 正常", "synonyms": ["黑名单标识", "是否黑名单"]},
        {"field_name": "remark", "field_type": "VARCHAR(255)", "comment": "备注", "synonyms": ["备注", "说明"]},
        {"field_name": "create_time", "field_type": "DATETIME", "comment": "记录创建时间", "synonyms": ["创建时间", "入库时间"]},
        {"field_name": "update_time", "field_type": "DATETIME", "comment": "记录更新时间", "synonyms": ["更新时间", "修改时间"]},
    ],
    "transaction_record": [
        {"field_name": "trans_no", "field_type": "VARCHAR(32)", "comment": "交易流水号（唯一）", "synonyms": ["流水号", "交易编号", "成交编号"]},
        {"field_name": "customer_no", "field_type": "VARCHAR(32)", "comment": "客户号", "synonyms": ["客户编号", "客户ID", "客户"]},
        {"field_name": "account_no", "field_type": "VARCHAR(32)", "comment": "资金账号", "synonyms": ["资金账户", "交易账号", "资金号"]},
        {"field_name": "security_code", "field_type": "VARCHAR(16)", "comment": "证券代码", "synonyms": ["证券编号", "股票代码", "产品代码", "代码"]},
        {"field_name": "security_name", "field_type": "VARCHAR(64)", "comment": "证券名称", "synonyms": ["股票名称", "产品名称", "证券简称"]},
        {"field_name": "security_type", "field_type": "VARCHAR(16)", "comment": "证券类型：股票/基金/债券/理财", "synonyms": ["品种", "证券类别", "资产类别"]},
        {"field_name": "market", "field_type": "VARCHAR(16)", "comment": "交易市场：上交所/深交所/北交所", "synonyms": ["交易所", "市场代码", "市场"]},
        {"field_name": "trade_direction", "field_type": "VARCHAR(8)", "comment": "交易方向：买入/卖出", "synonyms": ["方向", "买卖方向", "操作类型", "买/卖"]},
        {"field_name": "order_type", "field_type": "VARCHAR(8)", "comment": "委托类型：限价/市价", "synonyms": ["订单类型", "委托方式"]},
        {"field_name": "trade_price", "field_type": "DECIMAL(12,4)", "comment": "成交价格", "synonyms": ["成交价", "价格", "成交单价"]},
        {"field_name": "trade_quantity", "field_type": "BIGINT", "comment": "成交数量（股/份）", "synonyms": ["数量", "成交数量", "股数", "份额"]},
        {"field_name": "trade_amount", "field_type": "DECIMAL(20,2)", "comment": "成交金额", "calibre": "成交价 × 成交数量，含税、计手续费前", "synonyms": ["金额", "成交额", "交易金额", "成交金额"]},
        {"field_name": "commission", "field_type": "DECIMAL(12,2)", "comment": "手续费/佣金", "synonyms": ["佣金", "交易费用", "手续费"]},
        {"field_name": "stamp_duty", "field_type": "DECIMAL(12,2)", "comment": "印花税", "synonyms": ["税费", "印花税"]},
        {"field_name": "transfer_fee", "field_type": "DECIMAL(12,2)", "comment": "过户费", "synonyms": ["过户费", "结算费"]},
        {"field_name": "settle_amount", "field_type": "DECIMAL(20,2)", "comment": "清算/实际划转金额", "calibre": "成交金额 +/- 各项费用后的实际划转额", "synonyms": ["结算金额", "实际成交金额", "清算金额"]},
        {"field_name": "position_code", "field_type": "VARCHAR(32)", "comment": "持仓编号", "synonyms": ["持仓", "持仓序号"]},
        {"field_name": "fee_rate", "field_type": "DECIMAL(8,4)", "comment": "佣金费率（‰）", "synonyms": ["费率", "佣金费率"]},
        {"field_name": "trans_time", "field_type": "DATETIME", "comment": "成交时间", "synonyms": ["成交时间", "交易时间", "委托时间"]},
        {"field_name": "order_no", "field_type": "VARCHAR(32)", "comment": "委托编号", "synonyms": ["委托单号", "委托流水", "委托号"]},
        {"field_name": "trade_type", "field_type": "VARCHAR(16)", "comment": "交易类型：普通买卖/融资融券/转托管", "synonyms": ["业务类型", "交易类别"]},
        {"field_name": "margin_flag", "field_type": "TINYINT", "comment": "融资融券标记：1 信用交易 / 0 普通", "synonyms": ["信用交易标记", "两融标记"]},
        {"field_name": "counterparty", "field_type": "VARCHAR(64)", "comment": "对手方/对手机构", "synonyms": ["对手机构", "对手方"]},
        {"field_name": "venue", "field_type": "VARCHAR(16)", "comment": "交易场所/通道", "synonyms": ["场所", "交易通道"]},
        {"field_name": "t_plus_rule", "field_type": "VARCHAR(8)", "comment": "T+N 交收规则", "synonyms": ["交割规则", "交收规则", "T+N"]},
        {"field_name": "settle_date", "field_type": "DATE", "comment": "交割/交收日期", "synonyms": ["交收日期", "清算日期", "交割日"]},
        {"field_name": "status", "field_type": "VARCHAR(16)", "comment": "交易状态：已成交/部分成交/已撤销/待清算", "synonyms": ["状态", "成交状态"]},
        {"field_name": "channel", "field_type": "VARCHAR(16)", "comment": "委托渠道：网上/手机/柜台/程序化", "synonyms": ["渠道", "交易渠道", "委托渠道"]},
        {"field_name": "client_ip", "field_type": "VARCHAR(64)", "comment": "客户端 IP", "is_sensitive": True, "synonyms": ["IP", "客户端地址"]},
        {"field_name": "device_id", "field_type": "VARCHAR(64)", "comment": "设备标识", "is_sensitive": True, "synonyms": ["设备号", "设备ID"]},
        {"field_name": "remark", "field_type": "VARCHAR(255)", "comment": "备注", "synonyms": ["备注", "说明"]},
        {"field_name": "create_time", "field_type": "DATETIME", "comment": "记录创建时间", "synonyms": ["创建时间", "入库时间"]},
        {"field_name": "update_time", "field_type": "DATETIME", "comment": "记录更新时间", "synonyms": ["更新时间", "修改时间"]},
    ],
}


def import_dictionary() -> dict[str, int]:
    """导入示例字典，返回统计 {tables, fields}。幂等可重复执行。"""
    init_db()
    db = SessionLocal()
    stats = {"tables": 0, "fields": 0}
    try:
        for t in TABLES:
            table_row = db.query(DictTable).filter_by(table_name=t["table_name"]).first()
            if table_row is None:
                table_row = DictTable(**t)
                db.add(table_row)
                db.flush()
                stats["tables"] += 1
            else:
                for key, value in t.items():
                    setattr(table_row, key, value)

            for f in FIELDS[t["table_name"]]:
                existing = (
                    db.query(DictField)
                    .filter_by(table_id=table_row.id, field_name=f["field_name"])
                    .first()
                )
                if existing is None:
                    db.add(DictField(table_id=table_row.id, **f))
                    stats["fields"] += 1
                else:
                    for key, value in f.items():
                        setattr(existing, key, value)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    logger.info("dictionary_imported", tables=stats["tables"], fields=stats["fields"])
    return stats


if __name__ == "__main__":
    stats = import_dictionary()
    print(f"导入完成：新增表 {stats['tables']} 张，新增字段 {stats['fields']} 个（已存在的自动更新）。")
