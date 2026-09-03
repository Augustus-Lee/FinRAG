-- FinRAG 业务数据初始化（NL2SQL 智能问数的数据源）
-- MySQL 容器首次启动时自动执行（docker-entrypoint-initdb.d）
-- 创建 business 库 + 3 张业务表 + 示例数据

CREATE DATABASE IF NOT EXISTS business CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE business;

-- -----------------------------------------------------------------------
-- 1. 产品销售表（product_sales）
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_sales (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    trade_date   DATE         NOT NULL COMMENT '交易日期',
    product_name VARCHAR(128) NOT NULL COMMENT '产品名称',
    sales_amount DECIMAL(18,2) NOT NULL COMMENT '销售金额（元）',
    commission   DECIMAL(18,2) NOT NULL COMMENT '手续费（元）',
    risk_level   VARCHAR(8)   NOT NULL DEFAULT '中' COMMENT '风险等级（低/中/高）',
    PRIMARY KEY (id),
    INDEX idx_trade_date (trade_date),
    INDEX idx_product_name (product_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品销售明细表';

-- -----------------------------------------------------------------------
-- 2. 客户账户表（customer_account）
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customer_account (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    customer_no  VARCHAR(32)  NOT NULL COMMENT '客户编号',
    customer_name VARCHAR(64) NOT NULL COMMENT '客户姓名',
    mobile       VARCHAR(16)  NOT NULL COMMENT '手机号',
    id_card      VARCHAR(18)  NOT NULL COMMENT '身份证号',
    email        VARCHAR(64)  DEFAULT NULL COMMENT '电子邮箱',
    total_assets DECIMAL(18,2) NOT NULL DEFAULT 0.00 COMMENT '总资产（元）',
    open_date    DATE         NOT NULL COMMENT '开户日期',
    PRIMARY KEY (id),
    UNIQUE KEY uk_customer_no (customer_no),
    INDEX idx_customer_name (customer_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户账户信息表';

-- -----------------------------------------------------------------------
-- 3. 交易记录表（transaction_record）
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transaction_record (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    trade_date      DATE         NOT NULL COMMENT '交易日期',
    customer_no     VARCHAR(32)  NOT NULL COMMENT '客户编号',
    product_name    VARCHAR(128) NOT NULL COMMENT '产品名称',
    trade_amount    DECIMAL(18,2) NOT NULL COMMENT '成交金额（元）',
    commission      DECIMAL(18,2) NOT NULL COMMENT '手续费（元）',
    trade_type      VARCHAR(8)   NOT NULL COMMENT '交易类型（买入/卖出）',
    PRIMARY KEY (id),
    INDEX idx_trade_date (trade_date),
    INDEX idx_customer_no (customer_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交易记录明细表';

-- -----------------------------------------------------------------------
-- 示例数据
-- -----------------------------------------------------------------------
INSERT INTO product_sales (trade_date, product_name, sales_amount, commission, risk_level) VALUES
('2026-01-15', '稳健成长型基金A', 1500000.00, 4500.00, '低'),
('2026-02-20', '高风险股票型基金B', 3200000.00, 16000.00, '高'),
('2026-03-10', '国债逆回购7天', 800000.00, 800.00, '低'),
('2026-04-05', '黄金ETF', 2200000.00, 6600.00, '中'),
('2026-05-18', '稳健成长型基金A', 1800000.00, 5400.00, '低'),
('2026-06-22', '高风险股票型基金B', 4500000.00, 22500.00, '高'),
('2026-07-01', '可转债C', 1200000.00, 3600.00, '中'),
('2026-07-15', '黄金ETF', 3100000.00, 9300.00, '中');

INSERT INTO customer_account (customer_no, customer_name, mobile, id_card, email, total_assets, open_date) VALUES
('C20260001', '张三', '13800138001', '110101199001011234', 'zhangsan@example.com', 580000.00, '2024-03-15'),
('C20260002', '李四', '13900139002', '310101198505052345', 'lisi@example.com', 1250000.00, '2023-08-20'),
('C20260003', '王五', '13700137003', '440101199212123456', 'wangwu@example.com', 3200000.00, '2022-01-10'),
('C20260004', '赵六', '13600136004', '510101198808084567', 'zhaoliu@example.com', 750000.00, '2024-11-05'),
('C20260005', '钱七', '13500135005', '320101199506065678', 'qianqi@example.com', 2100000.00, '2023-06-18');

INSERT INTO transaction_record (trade_date, customer_no, product_name, trade_amount, commission, trade_type) VALUES
('2026-01-15', 'C20260001', '稳健成长型基金A', 100000.00, 300.00, '买入'),
('2026-02-20', 'C20260003', '高风险股票型基金B', 500000.00, 2500.00, '买入'),
('2026-03-10', 'C20260002', '国债逆回购7天', 200000.00, 200.00, '买入'),
('2026-04-05', 'C20260005', '黄金ETF', 300000.00, 900.00, '买入'),
('2026-05-18', 'C20260001', '稳健成长型基金A', 80000.00, 240.00, '卖出'),
('2026-06-22', 'C20260003', '高风险股票型基金B', 600000.00, 3000.00, '买入'),
('2026-07-01', 'C20260002', '可转债C', 150000.00, 450.00, '买入'),
('2026-07-15', 'C20260005', '黄金ETF', 250000.00, 750.00, '卖出');
