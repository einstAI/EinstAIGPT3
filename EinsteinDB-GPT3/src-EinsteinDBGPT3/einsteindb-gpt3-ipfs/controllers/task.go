package controllers

import (
	"fmt"
	"strings"
	_ "time"

	"github.com/astaxie/beego" // "github.com/astaxie/beego/orm"
	_ "github.com/go-sql-driver/mysql"

	"github.com/astaxie/beego/logs"
	"github.com/astaxie/beego/orm"
	"github.com/astaxie/beego/validation"

	"einsteindb-gpt3-ipfs/base"
)

type TaskController struct {
	beego.Controller
	Data interface{}
}

func (t *TaskController) SetData(data interface{}) {
	t.Data["json"] = data

}

func (t *TaskController) ServeJSON() {
	t.Ctx.Output.JSON(t.Data, true, true)
	t.ServeJSON()
}

func (t *TaskController) Post() {
	fmt.Println("task:::", t.Ctx.Request.URL)
	reqURL := strings.TrimPrefix(fmt.Sprint(t.Ctx.Request.URL), "")
	switch reqURL {
	case "/task/query_task_result":
		t.HandleQueryTaskResult()
	case "/task/create_task":
		// dapp.HandleQueryTask(w, r)
	case "/update_task":
		// dapp.HandleUpdateTask(w, r)
	case "/query_task_result":
		// dapp.HandleQueryTaskResult(w, r)
	case "/insert_task_result":
		// dapp.HandleInsertTaskResult(w, r)
	}

}

func (t *TaskController) Get() {
	fmt.Println("task:::", t.Ctx.Request.URL)
	reqURL := strings.TrimPrefix(fmt.Sprint(t.Ctx.Request.URL), "")
	switch reqURL {
	case "/task/query_task_result":
		t.HandleQueryTaskResult()
	case "/task/create_task":
		// dapp.HandleQueryTask(w, r)
	case "/update_task":
		// dapp.HandleUpdateTask(w, r)
	case "/query_task_result":
		// dapp.HandleQueryTaskResult(w, r)
	case "/insert_task_result":
		// dapp.HandleInsertTaskResult(w, r)
	}

}

func (t *TaskController) HandleQueryTaskResult() {
	bytes, err := base.HttpRedirectPost(remote_add+"/query_task_result", t.Ctx.Request)
	if err == nil {
		t.SetData(string(*bytes))
		t.ServeJSON()
	}
	TLog.Errorf("HandleQueryTaskResult err=%+v", err)
}
