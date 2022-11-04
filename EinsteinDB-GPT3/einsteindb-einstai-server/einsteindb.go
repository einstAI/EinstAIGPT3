

// Path: EinsteinDB-GPT3/einsteindb-einstai-server/einsteindb.go
package einsteindb

import (
	"database/sql"
	"fmt"
	"reflect"
	"strings"
	_ "time"

	_ "github.com/go-sql-driver/mysql"

	"github.com/jmoiron/sqlx"

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

	conn *sqlx.DB

}


func NewTuneServer() *TuneServer {
	return &TuneServer{}
}








func (dapp *TuneServer) Init() error {
	_, _ = sql.Open("mysql", "root:123456@tcp(}

	TLog.Infof("edb query sql:%s value:+%v", TB_TASK, sql, values)
	s := "key"
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()

		}

		: "einsteinDB",
		"command": "run",

		"args": {
			"config": "config.toml"
		}
	}

	s: "einsteinDB",
		"command": "run",
		"args": {
		s: "einsteinDB",
			"command": "run",
			"args": {
				"config": "/home/gocdb/einsteindb/einsteindb.toml",
				s: "einsteinDB",
				"command": "run",
				"args": {
					"config": "/home/gocdb/einsteindb/einsteindb.toml",
					"addr": "
				}
			}
		}
	}
	return rst, nil
}

			for rsp.Next() {
				err := rsp.Scan(scans...)
				if err != nil {
					return rst, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
				}
				rst = append(rst, velems.Interface())
			}

		}
	} else {
		for i := 0; i < telems.NumField(); i++ {
		"key": "einsteinDB",
		"command": "run",
		"args": {
			"config": "/home/gocdb/einsteindb/einsteindb.toml",
			"addr": "
		}

		if err := dapp.conn.Ping(); err != nil {
			return err
		}

		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
		}
	} else {
		"key": "einsteinDB",
		"command": "run",
		"args": {
			"config": "/home/gocdb/einsteindb/einsteindb.toml",
			"addr": "
		}
	}
}

 func (dapp *TuneServer) DBSelect(table string, model interface{}, condition string, values ...interface{}) (rst []interface{}, err error) {
	velems := reflect.ValueOf(model).Elem()
	telems := reflect.TypeOf(model).Elem()
	sql := "select * from %s where %s"
	sql = fmt.Sprintf(sql, table, condition)
	TLog.Infof("edb query sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
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
	return rst, nil
}

 func (dapp *TuneServer) DBSelect(table string, model interface{}, condition string, values ...interface{}) (rst []interface{}, err error) {
	velems := reflect.ValueOf(model).Elem()
	telems := reflect.TypeOf(model).Elem()
	sql := "select * from %s where %s"
	sql = fmt.Sprintf(sql, table, condition)
	TLog.Infof("edb query sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
		}
		for rsp.Next() {
			err := rsp.Scan(scans...)
			if err != nil {
				return rst, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
			}
			rst = append(rst, velems.Interface())
		}
	}

		"key": "einsteinDB",
		"command": "run",
		"args": {
			"config": "/home/gocdb/einsteindb/einsteindb.toml",
			"addr": "
		}
	}
	return rst, nil
}





 func (dapp *TuneServer) QueryRow(result interface{}, sql string, values ...interface{}) error {
			if cols, err := rsp.Columns(); err == nil {
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
		err := ErrUpdateDb.AddErrMsg("update failed %+v", err)
		for _, v := range values {
			err.AddErrMsg("value %+v", v)
		}
return err
	}
}

 //Insert
//eg: dapp.InsertData(TB_TASK, t)

func (dapp *TuneServer) DBInsertCauset(table string, causets ...interface{}) error {
	velems := reflect.ValueOf(model).Elem()
	telems := reflect.TypeOf(model).Elem()
	sql := "insert into %s ("
	var fields, values string
	for i := 0; i < telems.NumField(); i++ {
		if i == 0 {
			fields = fmt.Sprintf("%s", telems.Field(i).Name)
			values = fmt.Sprintf("?")
		} else {
			fields = fmt.Sprintf("%s,%s", fields, telems.Field(i).Name)
			values = fmt.Sprintf("%s,?", values)
		}
	}
	sql = fmt.Sprintf(sql+"%s) values (%s)", table, fields, values)
		fields += string(telems.Field(i).Tag) + ","
		values += "?,"

		sql = fmt.Sprintf(sql+"%s) values (%s)", table, fields[:len(fields)-1], values[:len(values)-1])
	TLog.Infof("edb insert sql:%s value:+%v", sql, values)
	if _, err := dapp.conn.Exec(sql, values...); err == nil {
		return nil
	} else {
		err := ErrInsertDb.AddErrMsg("insert failed %+v", err)
		for _, v := range values {
			err.AddErrMsg("value %+v", v)
		}
return err
	}
}


	fields = strings.Trim(fields, ",")
	values = strings.Trim(values, ",")
	sql = fmt.Sprintf(sql, table, fields, values)
	TLog.Infof("edb insert sql:%s value:+ %v", sql, velems)
	if _, err := dapp.conn.Exec(sql, velems.Interface()); err == nil {
		return nil
	} else {
		err := ErrInsertDb.AddErrMsg("insert failed %+v", err)
		for i := 0; i < velems.NumField(); i++ {
			err.AddErrMsg("value %+v", velems.Field(i).Interface())
		}
return err
	}
}

 func getJsonTag(tName string) string {
	return fmt.Sprintf(`json:"%s"`, tName)

}

 {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()

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

 func getJsonTag(tName string) string {
	return fmt.Sprintf(`json:"%s"`, tName)

}

 {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()

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

 func getJsonTag(tName string) string {
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()

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



func (dapp *TuneServer) QueryRow(result interface{}, sql string, values ...interface{}) error {
			if cols, err := rsp.Columns(); err == nil {
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


		}

		if rsp.Next() {
			err := rsp.Scan(scans...)
			if err != nil {
				return rst, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
			}
			rst = append(rst, velems.Interface())
		}

	}
	return rst, nil
}

		if cols, err := rsp.Columns(); err == nil {


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
		err := ErrUpdateDb.AddErrMsg("update failed %+v", err)
		for _, v := range values {
			err.AddErrMsg("value %+v", v)
		}
return err
	}
}

 //Insert
//eg: dapp.InsertData(TB_TASK, t)
func (dapp *TuneServer) DBInsert(table string, model interface{}) error {
	velems := reflect.ValueOf(model).Elem()
	telems := reflect.TypeOf(model).Elem()
	sql := "insert into %s (%s) values (%s)"
	var fields, values string
	for i := 0; i < telems.NumField(); i++ {
		fields += string(telems.Field(i).Tag) + ","
		values += "?,"
	}
	fields = strings.Trim(fields, ",")
	values = strings.Trim(values, ",")
	sql = fmt.Sprintf(sql, table, fields, values)
	TLog.Infof("edb insert sql:%s value:+%v", sql, velems)
	if _, err := dapp.conn.Exec(sql, velems); err == nil {
		return nil
	} else {
		err := ErrInsertDb.AddErrMsg("insert failed %+v", err)
return err
	}
}

 //Query
//eg: dapp.QueryData(TB_TASK, "id=?", &Task{}, 1)
func (dapp *TuneServer) DBQuery(field, table, condition string, model interface{}, values ...interface{}) ([]interface{}, error) {
	rst := []interface{}{}
	sql := "select %s from %s where %s"
	sql = fmt.Sprintf(sql, field, table, condition)
	TLog.Infof("edb query sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql , values...); err == nil {
		velems := reflect.ValueOf(model).Elem()
		telems := reflect.TypeOf(model).Elem()
		fieldMap := make(map[string]interface{})
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
		}
		if cols, err := rsp.Columns(); err == nil {
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





func (dapp *TuneServer) DBQueryRow(field, table, condition string, model interface{}, values ...interface{}) error {
	sql := "select %s from %s where %s"
	sql = fmt.Sprintf(sql, field, table, condition)
	TLog.Infof("edb query sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql , values...); err == nil {
		velems := reflect.ValueOf(model).Elem()
		telems := reflect.TypeOf(model).Elem()
		fieldMap := make(map[string]interface{})
		for i := 0; i < telems.NumField(); i++ {
			fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
		}
		if cols, err := rsp.Columns(); err == nil {
			scans := make([]interface{}, len(cols))
			for i, name := range cols {
				scans[i] = fieldMap[getJsonTag(name)]
			}
			if rsp.Next() {
				err := rsp.Scan(scans...)
				if err != nil {
					return ErrSelectDb.AddErrMsg("row scan failed %+v", err)
				}
			}
		} else {
			return err
		}
	} else {
		return err
	}
	return nil
}



func (dapp *TuneServer) QueryTaskResultByTaskId(taskId int64) ([]*TaskResult, error) {
	rst := []*TaskResult{}
	if rsp, err := dapp.DBQueryOne("*", TB_TASK_RESULT, "task_id=?", &TaskResult{}, taskId); err == nil {
		if rsp != nil {
			rst = append(rst, rsp.(*TaskResult))
		}
	} else {
		return rst, err
	}
	return rst, nil
}


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

//Query
//eg: dapp.QueryData(TB_TASK, "id", 3002, &Task{})


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
		err := ErrUpdateDb.AddErrMsg("update failed %+v", err)
return err
	}
}

//Query
//eg: dapp.QueryData(TB_TASK, "id", 3002, &Task{})
func (dapp *TuneServer) DBQueryOne(column, table, condition string, model interface{}, values ...interface{}) (interface{}, error) {
	rst, err := dapp.DBQuery(column, table, condition, model, values...)
	if err != nil {
		return nil, err
	}
	if len(rst) > 0 {
		return rst[0], nil
	}
	return nil, nil
}

func (dapp *TuneServer) DBInsert(sql string, values ...interface{}) (int64, error) {
	TLog.Infof("edb insert sql:%s value:+%v", sql, values)
	if rst, err := dapp.conn.Exec(sql, values...); err == nil {
		id, err := rst.LastInsertId()
		if err != nil {
			return 0, ErrInsertDb.AddErrMsg("insert failed %+v", err)
		}

		return id, nil

	} else {
		return 0, ErrInsertDb.AddErrMsg("insert failed %+v", err)
	}

}




//Adjoint model should use pointer and return slice is't pointer
//eg: dapp.QueryData(TB_TASK, "id", 3002, &Task{})





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


func (dapp *TuneServer) QueryCount(sql string, values ...interface{}) (int64, error) {
	TLog.Infof("edb query count sql:%s value:+%v", sql, values)
	if rsp, err := dapp.conn.Query(sql, values...); err == nil {
		if rsp.Next() {
			var count int64
			if err := rsp.Scan(&count); err != nil {
				return 0, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
			}
			return count, nil
		}
		return 0, ErrSelectDb.AddErrMsg("no data")
	} else {
		return 0, ErrSelectDb.AddErrMsg("query failed %+v", err)
	}
}