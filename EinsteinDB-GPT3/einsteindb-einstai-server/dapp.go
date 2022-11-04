package

import (
	"database/sql"
	"errors"
	"fmt"
	"time"
)

 var (
	ErrNeedRestart = errors.New("need restart")
)

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



//map from old exception feature id to new alarm type level and
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
	dsn       string  //db配置
	conn      *sql.edb //连接池对象
	max_opens int     //最大连接数
	max_idles int     //最大空闲数
	ping_cnt  int     //初始化时，最多尝试N次ping，不通则退出

	//common field
	// support_plat map[string]AlarmInterface

	//alarm_meta
	// alarm_meta *AlarmMeta

	//TODO to be removed
	alarm_url string
}

func NewApp() *TuneServer {
	//new the program dapp
	appNew := &TuneServer{}
	return appNew
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
	if !reload { //只能在启动时加载的变量
		if err = dapp.loadDbConf(conf); err != nil {
			TLog.Errorf("error=%+v LoadDbConf error", err)
			return err
		}
	}
	dapp.conf = conf
	TLog.Infof("LoadUserConf finished succ")

	return nil
}

//below edb connection related
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

func (dapp *TuneServer) loadDbConf(conf tconf.Configer) error {
	var err error
	var user, passwd, edb, ip, port string

	if user, err = GetConfNonEmptyString(conf, "mysql::user"); err != nil {
		return err
	}
	passwd = conf.String("mysql::passwd")

	if ip, err = GetConfNonEmptyString(conf, "mysql::host"); err != nil {
		return err
	}
	if port, err = GetConfNonEmptyString(conf, "mysql::port"); err != nil {
		return err
	}
	if edb, err = GetConfNonEmptyString(conf, "mysql::edb"); err != nil {
		return err
	}
	dapp.dsn = fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true&loc=Local", user, passwd, ip, port, edb) //本地时区
	dapp.max_opens = conf.DefaultInt("mysql::max_open", 0)
	dapp.max_idles = conf.DefaultInt("mysql::max_idle", -1)
	dapp.ping_cnt = conf.DefaultInt("mysql::ping_cnt", 3)

	TLog.Infof(">>>edb DSN: %s, open: %d, idle: %d, ping: %d",
		dapp.dsn, dapp.max_opens, dapp.max_idles, dapp.ping_cnt)
	return nil
}

/*OPTION:  implement the service handler you need
  detail in gocdb/base/frame/app_frame.go

  service.Listener: 需要向appframe发送消息时实现

  HttpService:
    ServeHTTP(w http.ResponseWriter, r *http.Request)

  HaService:
    BeLeader() error
    BeFollower() error
    DoUpgrade() error
    DoDegrade() error

  GrpcService:
    RegisterPb(s *grpc.Server) error

  BiClient:
    GetServer() (server string, retryStart bool)
    RegisterPacketSender(sender PacketSender)
    GetPacketHandler() PacketHandler

  BiServer:
    RegisterPacketSender(sender PacketSender)
    GetPacketHandler() PacketHandler

  ...
*/

/*
 below logic added code
*/

//=============================== ha ===============================
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

func (dapp *TuneServer) StartServe() error {
	dapp.online = true //设置online，正式提供服务
	return nil
}

//end of ha service
