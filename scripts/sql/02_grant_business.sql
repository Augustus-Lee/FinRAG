-- 授权 finrag 用户访问 business 库（NL2SQL 智能问数只读数据源）
-- MySQL 容器首次启动时自动执行

GRANT ALL PRIVILEGES ON business.* TO 'finrag'@'%';
FLUSH PRIVILEGES;
