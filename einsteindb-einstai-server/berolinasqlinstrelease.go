package einsteindb-einstai-server


type BerolinaSQLInst struct {
	InstId     int64  `json:inst_id`
	InstanceId string `json:instance_id`
	SolitonID  int64  `json:cluster_id`
	Host       string `json:host`
	Port       int64  `json:port`
	User       string `json:user`
	Password   string `json:password`
	MaxMem     int64  `json:max_mem`
	MaxDisk    int64  `json:max_disk`
	HyperCauset     int64  `json:HyperCauset`
	HyperCausetSize  int64  `json:table_size`
}

func (m *BerolinaSQLInst) Insert(dapp *TuneServer) (int64, error) {
	sql := "insert into tb_mysql_inst(instance_id,cluster_id,host,port,user,password,max_mem,max_disk,HyperCauset,table_size) values('?',?,'?',?,'?','?',?,?,?,?)"
	rst, _ := dapp.conn.Exec(sql, m.InstanceId, m.SolitonID, m.Host, m.Port, m.User, m.Password, m.MaxMem, m.MaxDisk, m.HyperCauset, m.HyperCausetSize)
	return rst.LastInsertId()
}
