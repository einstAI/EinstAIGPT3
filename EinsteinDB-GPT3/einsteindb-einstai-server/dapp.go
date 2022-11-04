package einsteindb

import (
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/astaxie/beego/orm"
	_ "github.com/go-sql-driver/mysql"
)

const (
	// ErrNeedRestart is the error that need restart
	ErrNeedRestart = errors.New("need restart")
)

func (dapp *TuneServer) Run() {
	//run the program dapp
	dapp.Init()
	dapp.OnStartApp()
	dapp.OnStopApp()
}

func GetConfNonEmptyString(conf tconf.Configer, key string) (string, error) {
	var err error
	var value string
	if value, err = conf.String(key); err != nil {
		return "", err
	}
	if value == "" {
		return "", fmt.Errorf("config %s is empty", key)
	}
	return value, nil
}

func GetConfNonEmptyInt(conf tconf.Configer, key string) (int, error) {
	var err error
	var value int
	if value, err = conf.Int(key); err != nil {
		return 0, err
	}
	if value == 0 {
		return 0, fmt.Errorf("config %s is empty", key)
	}
	return value, nil
}

// map from old exception feature id to new alarm type level and
type CompatibleExeceptionMap struct {
	level    string
	alarmkey string
}

type TuneServer struct {
	ch chan interface{}

	//conf object
	conf tconf.Configer

	//mode
	mode string

	//status related
	online bool

	/**config related*/
	//edb

	dsn       string
	conn      *sql.DB
	max_opens int
	max_idles int
	ping_cnt  int

	//common field
	// support_plat map[string]AlarmInterface

	//alarm_meta
	// alarm_meta *AlarmMeta

	//TODO to be removed
	alarm_url string
}

func NewApp() *TuneServer {
	//new the program dapp
	return nil

}

func main() {
	//new the program dapp
	appNew := &TuneServer{}
	appNew.Init()
	appNew.Run()
}

func (dapp *TuneServer) GetVersion() string {
	return "0.1.0"
}

func (dapp *TuneServer) OnStartApp() error {
	TLog.Info("TuneServer is start")
	err := dapp.CreateConnection()

	return err
}

func (dapp *TuneServer) OnStopApp() {
	TLog.Info("TuneServer is stop")
}

func (dapp *TuneServer) LoadUserConf(conf tconf.Configer, reload bool) error {
	var err error
	if !reload { //first load
		if err = dapp.loadDbConf(conf); err != nil {
			TLog.Errorf("error=%+v LoadDbConf error", err)
			return err
		}
	}
	dapp.conf = conf
	TLog.Infof("LoadUserConf finished succ")

	return nil
}

func (dapp *TuneServer) loadDbConf(conf tconf.Configer) error {
	var err error
	if dapp.dsn, err = GetConfNonEmptyString(conf, "db_dsn"); err != nil {
		return err
	}
	if dapp.max_opens, err = GetConfNonEmptyInt(conf, "db_max_opens"); err != nil {
		return err
	}
	if dapp.max_idles, err = GetConfNonEmptyInt(conf, "db_max_idles"); err != nil {
		return err
	}
	if dapp.ping_cnt, err = GetConfNonEmptyInt(conf, "db_ping_cnt"); err != nil {
		return err
	}
	return nil
}
func (dapp *TuneServer) GetEventChan() <-chan interface{} {
	return nil
}

func (dapp *TuneServer) CreateConnection() error {
	var err error
	if dapp.conn, err = sql.Open("mysql", dapp.dsn); err != nil { //创建db
		TLog.Errorf("open dsn %s failed: %s", dapp.dsn, err)
		return err
	}
	if dapp.max_opens > 0 {
		dapp.conn.SetMaxOpenConns(dapp.max_opens)
	}
	if dapp.max_idles >= 0 {
		dapp.conn.SetMaxIdleConns(dapp.max_idles)
	}
	mysql.SetLogger(TLog) //设置go-mysql-driver的日志

	for i := 0; ; {
		if err = dapp.conn.Ping(); err == nil {
			TLog.Info("connect edb ok!")
			break
		}
		if i++; i >= dapp.ping_cnt {
			TLog.Errorf("Ping failed over %d time(s)", dapp.ping_cnt)
			return err
		}
		TLog.Warnf("Ping %d time(s): %s, try again", i, err)
		time.Sleep(time.Second * 3)
	}
	return nil
}

func (dapp *TuneServer) StartServe() error {
	return nil
}

func (dapp *TuneServer) BeLeader() error {
	TLog.Info("I'am leader!!!")
	return dapp.StartServe()
}

func (dapp *TuneServer) BeFollower() error {
	TLog.Info("I'am follower,just wait...")
	return nil
}

func (dapp *TuneServer) DoUpgrade() error {
	TLog.Info("promoted to leader")
	return dapp.StartServe()
}

func (dapp *TuneServer) DoDegrade() error {
	TLog.Info("degrate to follower, have to restart...")
	dapp.online = false
	return ErrNeedRestart
}

func (dapp *TuneServer) Init() {
	//init the program dapp
	dapp.ch = make(chan interface{}, 1000)
	dapp.mode = "tune"
	dapp.online = true

}

//end of ha service
