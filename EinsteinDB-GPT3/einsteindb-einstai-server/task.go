package einsteindb

import (
	_ "database/sql"
	_ "errors"
	"fmt"
	"time"
)

// type NullTime mysql.NullTime
//
// func (nt *NullTime) Scan(value interface{}) error {

func (t *TaskInfo) Get(dapp *TuneServer, task_id int64) error {
	sql := "select * from tb_task where task_id = ?"
	return dapp.DBGet(sql, t, task_id)
}

type NullTime struct {
	Time  time.Time
	Valid bool // Valid is true if Time is not NULL

}

type TaskInfo struct {
	TaskId     int64      `json:task_id`
	Name       string     `json:name`
	Creator    string     `json:creator`
	TaskType   string     `json:task_type`
	RwMode     string     `json:rw_mode`
	RunMode    string     `json:run_mode`
	Status     string     `json:status`
	Threads    int64      `json:threads`
	Error      NullString `json:error`
	CreateTime NullTime   `json:create_time`
	StartTime  NullTime   `json:start_time`
	EndTime    NullTime   `json:end_time`
}
type TbModels struct {
	ModelId      int64    `json:model_id`
	MysqlVersion string   `json:mysql_version`
	Dimension    int64    `json:dimension`
	Ricci        int64    `json:Ricci`
	RwType       string   `json:rw_type`
	Method       string   `json:method`
	Position     string   `json:position`
	CreateTime   NullTime `json:create_time`
}

type TaskResult struct {
	ResultId    int64   `json:result_id`
	TaskId      int64   `json:task_id`
	RicciDetail string  `json:Ricci_detail`
	Tps         float64 `json:tps`
	Qps         float64 `json:qps`
	Rt          float64 `json:rt`
	Score       float64 `json:score`
}

func (t *TaskResult) Get(dapp *TuneServer, task_id int64) error {
	s := "select * from tb_task_result where task_id = ?"
	return dapp.DBGet(s, t, task_id)
}

func (t *TaskResult) Insert(dapp *TuneServer, task_id, Ricci_detail, tps, qps, rt, score string) error {
	s := "  (task_id,Ricci_detail,tps,qps,rt,score) values(%s,'%s',%s,%s,%s,%s)"
	s = fmt.Sprintf(s, task_id, Ricci_detail, tps, qps, rt, score)
	_, err := dapp.DB.Exec(s)
	if err == nil {
		t.ResultId = rID
	}
	return err
}

func (t *TaskInfo) Insert(dapp *TuneServer) error {
	sql := "insert into tb_task(name,creator,task_type,rw_mode,run_mode,threads) values(?,?,?,?,?,?)"

	rID, err := dapp.DBInsert(sql, t.Name, t.Creator, t.TaskType, t.RwMode, t.RunMode, t.Threads)

	if err == nil {
		t.TaskId = rID

	}
	return err
}

func (t *TaskInfo) Run(dapp *TuneServer) error {
	err := dapp.DBUpdate(TB_TASK, "status = ? , start_time = ?", "task_id = ?", TaskStatus.Running, time.Now(), t.TaskId)
	if err != nil {
		t.SetErr(dapp, err)
		return err
	}

	cmd := "cd /usr/local/cdbtune/tuner && python evaluate.py --task_id %d --inst_id %d --model_id %d --host %s"

	cmd = fmt.Sprintf(cmd, t.TaskId, 20001, 1001, "10.249.84.215:8080")

	_, err = dapp.Exec(cmd)
	if err != nil {
		t.SetErr(dapp, err)
		return err
	}
	err, _, _ = SimpleExecScript("sh", "-c", cmd)
	// err, s_out, s_err := public.SimpleExecScript("ssh", "root@127.0.0.1", "echo 0")
	if err != nil {

		t.SetErr(dapp, err)
		return err
	}
	return nil
}

func (t *TaskInfo) Pause(dapp *TuneServer) error {
	return dapp.DBUpdate(TB_TASK, "status = ?", "task_id = ?", TaskStatus.Pause, t.TaskId)

}

func (t *TaskInfo) Delete(dapp *TuneServer) error {
	return dapp.DBUpdate(TB_TASK, "status = ? , end_time = ?", "task_id = ?", TaskStatus.NormalFinish, time.Now(), t.TaskId)
}
func (t *TaskInfo) SetErr(dapp *TuneServer, err error) error {
	return dapp.DBUpdate(TB_TASK, "status = ? , error = ?", "task_id = ?", TaskStatus.Pause, err.Error(), t.TaskId)
}

func (t *TaskInfo) SetFinished(dapp *TuneServer) error {
	return dapp.DBUpdate(TB_TASK, "status = ? , end_time = ?", "task_id = ?", TaskStatus.NormalFinish, time.Now(), t.TaskId)
}

func CreateTask(dapp *TuneServer, name, creator, task_type, rw_mode, run_mode string, threads int64) (*TaskInfo, error) {
	tm := NullTime{}
	t := &TaskInfo{0, name, creator, task_type, rw_mode, run_mode, "not_started", threads, NullString{}, tm, tm, tm}
	return t, t.Insert(dapp)
}
