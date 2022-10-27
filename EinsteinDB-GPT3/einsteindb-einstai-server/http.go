package main

import (
	"net/http"
	"strconv"

	. "git.code.oa.com/gocdb/base/public"
)

type ProtCommonRsp struct {
	Errno int         `json:"errno"`
	Error string      `json:"error"`
	Data  interface{} `json:"data"`
}

func SendRsp(w http.ResponseWriter, data interface{}, err error) error {
	var rsp ProtCommonRsp

	if err == nil {
		rsp.Errno = 0
		rsp.Error = ""
		rsp.Data = data
	} else if e, ok := err.(*CDBError); ok {
		rsp.Errno = e.Errno()
		rsp.Error = e.Error()
		rsp.Data = data
	} else {
		rsp.Errno = ER_OUTER
		rsp.Error = err.Error()
		rsp.Data = data
	}
	err2 := SendHttpJsonRsp(w, rsp)
	if err2 != nil {
		TLog.Errorf("SendHttpJsonRsp failed +%v", err2)
	}
	return err2
}

//=============================== http ===============================
func (dapp *TuneServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	/*
		if !dapp.online { //服务未启动
			TLog.Error("http request arrived but server is not online!")
			var rsp ProtCommonRsp
			rsp.SendErrRsp(w, ErrServerOffline)
			return
		}
	*/
	TLog.Info(r.URL.Path)
	switch r.URL.Path {
	case "/create_task":
		dapp.HandleCreateTask(w, r)
	case "/query_task":
		dapp.HandleQueryTask(w, r)
	case "/update_task":
		dapp.HandleUpdateTask(w, r)
	case "/query_task_result":
		dapp.HandleQueryTaskResult(w, r)
	case "/insert_task_result":
		dapp.HandleInsertTaskResult(w, r)
	default:
		http.NotFound(w, r)
		return
	}
}

func hasErr(w http.ResponseWriter, err error) bool {
	if err != nil {
		err := SendRsp(w, nil, err)
		return err == nil
	}
	return false
}

func isEmpty(r *http.Request, fields ...string) (bool, error) {
	for _, v := range fields {
		if len(r.FormValue(v)) == 0 {
			TLog.Errorf("request [%s] form param empty", v)
			err := ErrHttpReq.AddErrMsg("request [%s] form param empty", v)
			return false, err
		}
	}
	return true, nil
}

func getIntField(r *http.Request, field string) (int64, error) {
	ret, err := strconv.Atoi(r.FormValue(field))
	if err != nil {
		TLog.Errorf("client request [%s] not int type err=%+v", field, err)
		return -1, ErrHttpReq.AddErrMsg("client request [%s] not int type err=%+v", field, err)
	}
	return int64(ret), nil
}

//create a new task
func (dapp *TuneServer) HandleCreateTask(w http.ResponseWriter, r *http.Request) {

	fields := []string{"name", "creator", "task_type", "rw_mode", "run_mode"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	ti, err := CreateTask(dapp,
		r.FormValue("name"),
		r.FormValue("creator"),
		r.FormValue("task_type"),
		r.FormValue("rw_mode"),
		r.FormValue("run_mode"),
		16,
	)
	if !hasErr(w, err) {
		SendRsp(w, ti, nil)
	}
	//TODO 基于channel 运行
	go ti.Run(dapp)
}
func (dapp *TuneServer) HandleQueryTask(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	// var timeBase time.Time
	if !hasErr(w, err) {
		tInfo := &TaskInfo{}
		err := dapp.QueryByIndex(TB_TASK, "task_id", tid, tInfo)
		if !hasErr(w, err) {
			SendRsp(w, tInfo, nil)
		}
	}
}
func (dapp *TuneServer) HandleUpdateTask(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	errMsg := r.FormValue("error")
	if !hasErr(w, err) {
		tInfo := &TaskInfo{}
		err := dapp.QueryByIndex(TB_TASK, "task_id", tid, tInfo)
		if !hasErr(w, err) {
			if len(errMsg) == 0 {
				err = tInfo.SetFinished(dapp)
			} else {
				err = tInfo.SetErr(dapp, ErrTuneFailed.AddErrMsg(errMsg))
			}
			if !hasErr(w, err) {
				SendRsp(w, tInfo, nil)
			}
		}
	}
}
func (dapp *TuneServer) HandleQueryTaskResult(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	if !hasErr(w, err) {
		tr := &TaskResult{}
		rst, err := dapp.DBQuery("*", TB_TASK_RESULT, "task_id = ?", tr, tid)
		if !hasErr(w, err) {
			SendRsp(w, rst, nil)
		}
	}
}
func (dapp *TuneServer) HandleInsertTaskResult(w http.ResponseWriter, r *http.Request) {
	fields := []string{"task_id", "Ricci_detail", "tps", "qps", "rt", "score"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	tr := &TaskResult{}
	err := tr.Insert(dapp,
		r.FormValue("task_id"),
		r.FormValue("Ricci_detail"),
		r.FormValue("tps"),
		r.FormValue("qps"),
		r.FormValue("rt"),
		r.FormValue("score"),
	)
	if !hasErr(w, err) {
		SendRsp(w, tr, nil)
	}
}

//end of http service
