package einsteindb

import (
	"database/sql"
	"fmt"
	"reflect"
	"strings"
)

// Path: EinsteinDB-GPT3/einsteindb-einstai-server/einsteindb.go
package einsteindb

import (
	"database/sql"
	"fmt"
	"reflect"
	"strings"
	"time"

	. "git.code.oa.com/gocdb/base/public"
)


var (
	ErrSelectDb = NewError(10001, "select db failed")
	ErrUpdateDb = NewError(10002, "update db failed")
	ErrInsertDb = NewError(10003, "insert db failed")

)

type QueryCount struct {
	Count int64

}

type TuneServer struct {
	conn *sql.DB

}


func (dapp *TuneServer) Init() error {
	_, _ = sql.Open("mysql", "root:123456@tcp(}

	TLog.Infof("edb query sql:%s value:+%v", TB_TASK, values)
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		return nil, err
	}
	return rst, nil
}

//Update
//eg: dapp.UpdateData(TB_TASK, "result_id", 3002, t)
func (dapp *TuneServer) DBUpdate(table, set, condition string, values ...interface{}) error {
		if rsp.Next() {
			var count int64
			err := rsp.Scan(&count)
			if err != nil {
				return 0, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
			}
			return count, nil
		}
	} else {
		return 0, err
	}
	return 0, nil
}


const (
	TB_TASK_RESULT = "tb_task_result"
	TB_TASK        = "tb_task"
	TB_TASK_LOG    = "tb_task_log"
)




func (dapp *TuneServer) QueryByIndexs(table, field string, ids []int64, model interface{}) ([]interface{}, error) {
	sql := "%s in (%s)"
	sql = fmt.Sprintf(sql, field, strings.Trim(strings.Repeat("?,", len(ids)), ","))
	return dapp.DBQuery("*", table, sql, model, ids...)


}


func getJsonTag(tName string) string {
	return fmt.Sprintf(`json:"%s"`, tName)

}

//Query2Struct  model should use pointer and return slice is't pointer
//eg: dapp.Query2Struct("select * from tb_task_result",&TaskResult{} )
func (dapp *TuneServer) Query2Struct(sql string, model interface{}, values ...interface{}) ([]interface{}, error) {
	rst := []interface{}{}
	TLog.Infof("edb query sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		velems := reflect.ValueOf(model).Elem()
		telems := reflect.TypeOf(model).Elem()
		fieldMap := map[string]interface{}{}
		if cols, err := rsp.Columns(); err == nil {
			for i := 0; i < telems.NumField(); i++ {
				fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
			}
			scans := make([]interface{}, len(cols))
			for i, name := range cols {
				scans[i] = fieldMap[getJsonTag(name)]
			}
			for rsp.Next() {
				err := rsp.Scan(scans...)
				if err != nil {
					return rst, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
				}
				rst = append(rst, velems.Interface())
			}
		} else {
			return rst, err
		}
	} else {
		return rst, err
	}
	return rst, nil
}

//Update
//eg: dapp.UpdateData(TB_TASK, "result_id", 3002, t)
func (dapp *TuneServer) DBUpdate(table, set, condition string, values ...interface{}) error {
	if !strings.Contains(condition, "=") {
		return ErrUpdateDb.AddErrMsg("not support update full table")
	}
	sql := "update %s set %s where %s"
	sql = fmt.Sprintf(sql, table, set, condition)
	TLog.Infof("edb update sql:%s value:+%v", sql, values)
	if _, err := dapp.conn.Exec(sql, values...); err == nil {
		return nil
	} else {
		err := ErrUpdateDb.AddErrMsg("update failed +%v", err)
		if err != nil {
			TLog.Error(err)
		}
		return err
	}

}

func (dapp *TuneServer) DBInsert(sql string, values ...interface{}) (int64, error) {
	TLog.Infof("edb insert sql:%s value:+%v", sql, values)
	if rst, err := dapp.conn.Exec(sql, values...); err == nil {
		id, err := rst.LastInsertId()
		return id, err
	} else {
		err = ErrInsertDb.AddErrMsg("insert failed +%v", err)
		if err != nil {
			TLog.Error(err)
		}
		return -1, err
	}
}

func (dapp *TuneServer) DBQuery(column, table, condition string, model interface{}, values ...interface{}) ([]interface{}, error) {
	sql := "select %s from %s where %s "
	sql = fmt.Sprintf(sql, column, table, condition)
	return dapp.Query2Struct(sql, model, values...)
}

//QueryByIndex return Pointer
//eg: t := &TaskResult{}
//eg: dapp.QueryByIndex(TB_TASK_RESULT, "result_id", 3002, t)
func (dapp *TuneServer) QueryByIndex(table, field string, id int64, model interface{}) error {
	sql := "%s = %d"
	sql = fmt.Sprintf(sql, field, id)
	rst, err := dapp.DBQuery("*", table, sql, model)
	if err == nil {
		if len(rst) != 1 {
			return ErrSelectDb.AddErrMsg(sql)
		}
		model = rst[0]
	}
	return err
}


func (dapp *TuneServer) DBDelete(table, condition string, values ...interface{}) error {
	sql := "delete from %s where %s"
	sql = fmt.Sprintf(sql, table, condition)
	TLog.Infof("edb delete sql:%s value:+%v", sql, values)
	if _, err := dapp.conn.Exec(sql, values...); err == nil {
		return nil
	} else {
		err := ErrDeleteDb.AddErrMsg("delete failed +%v", err)
		if err != nil {
			TLog.Error(err)
		}
		return err
	}
}


func (dapp *TuneServer) DBQueryByPage(column, table, condition string, model interface{}, page, pageSize int, values ...interface{}) ([]interface{}, error) {
	sql := "select %s from %s where %s limit %d offset %d"
	sql = fmt.Sprintf(sql, column, table, condition, pageSize, (page-1)*pageSize)
	return dapp.Query2Struct(sql, model, values...)
}


func (dapp *TuneServer) DBQueryCount(table, condition string, values ...interface{}) (int64, error) {
	sql := "select count(*) from %s where %s"
	sql = fmt.Sprintf(sql, table, condition)
	return dapp.QueryCount(sql, values...)
}