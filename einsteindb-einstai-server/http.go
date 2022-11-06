package einsteindb

import (
	"encoding/json"
	"net/http"
	"strconv"

	. "git.code.oa.com/gocdb/base/public/err"
	. "git.code.oa.com/gocdb/base/public/log"
	. "git.code.oa.com/gocdb/base/public/prot"
	. "git.code.oa.com/gocdb/base/public/prot/tune"
	. "git.code.oa.com/gocdb/base/public/prot/tune/task"
	. "git.code.oa.com/gocdb/base/public/prot/tune/task/result"

	. "git.code.oa.com/gocdb/base/public/prot/tune/task/result"
)

//CHANGELOG:  We have to use the same task_id to update the task

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
	} else if e, ok := err.(*Err); ok {
		rsp.Errno = e.Errno
		rsp.Error = e.Error()
		rsp.Data = data
	} else {
		rsp.Errno = ER_OUTER
		rsp.Error = err.Error()
		rsp.Data = data
	}
	err2 := SendJson(w, rsp)
	if err2 != nil {
		TLog.Errorf("SendHttpJsonRsp failed +%v", err2)
	}
	return err2

}

func (dapp *TuneServer) HandleQueryTask(w http.ResponseWriter, r *http.Request) {
	fields := []string{"task_id"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	task_id, err := getIntField(r, "task_id")
	if err != nil {
		SendRsp(w, nil, err)
		return
	}
	ti, err := QueryTask(dapp, task_id)
	if !hasErr(w, err) {
		SendRsp(w, ti, nil)
	}

}

func SendJson(w http.ResponseWriter, rsp ProtCommonRsp) interface{} {

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	return json.NewEncoder(w).Encode(rsp)
}

// =============================== http ===============================
func (dapp *TuneServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	/*
		if r.Method == "GET" {
			dapp.HandleGet(w, r)
		} else if r.Method == "POST" {
			dapp.HandlePost(w, r)
		} else {
			w.WriteHeader(http.StatusMethodNotAllowed)
		}



	*/

	//TODO
	//dapp.HandlePost(w, r)
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

func (dapp *TuneServer) HandleCreateTask(w http.ResponseWriter, r *http.Request) {
	fields := []string{"task_type", "task_param"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	task_type, err := getIntField(r, "task_type")
	if err != nil {
		SendRsp(w, nil, err)
		return
	}
	task_param := r.FormValue("task_param")
	task_id, err := CreateTask(dapp, task_type, task_param)
	if !hasErr(w, err) {
		SendRsp(w, task_id, nil)
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

// create a new task
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

	go ti.Run(dapp)
}
func (dapp *TuneServer) SolitonFilter(w http.ResponseWriter, r *http.Request) {
	fields := []string{"task_id", "sql"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	task_id, err := getIntField(r, "task_id")
	if err != nil {
		SendRsp(w, nil, err)
		return
	}
	sql := r.FormValue("sql")
	ti, err := QueryTask(dapp, task_id)
	if !hasErr(w, err) {
		SendRsp(w, ti, nil)
	}
	ti.SolitonFilter(dapp, sql)
}

// =============================== http ===============================

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
