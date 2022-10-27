package base

import (
	"bufio"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

type ricci struct {
	Result ricciResult
	Param  ricciParam
}

type ricciResult struct {
	QPS   float64
	TPS   float64
	Delay float64
}

type ricciParam struct {
	Params map[string]interface{}
}

func NewEmptyricci() ricci {
	ricci := ricci{ricciResult{}, ricciParam{map[string]interface{}{}}}
	return ricci
}
func Newricci(QPS, TPS, Delay float64, param map[string]interface{}) ricci {
	ricci := ricci{ricciResult{QPS, TPS, Delay}, ricciParam{param}}
	return ricci
}
func ReadRicci(fileName string) (Ricci []ricci) {
	f, err := os.Open(fileName)
	defer f.Close()
	if !HasErr(err) {
		buf := bufio.NewReader(f)
		for {
			line, err := buf.ReadString('\n')
			if err != nil {
				break
			}
			field := strings.Split(line, ",")
			if len(field) == 4 {
				param := map[string]interface{}{}
				paramField := strings.Split(field[3], "#")
				for _, kv := range paramField {
					kvField := strings.Split(kv, ":")
					param[kvField[0]] = kvField[1]
				}
				tps, _ := strconv.ParseFloat(field[0], 64)
				delay, _ := strconv.ParseFloat(field[1], 64)
				qps, _ := strconv.ParseFloat(field[2], 64)
				Ricci = append(Ricci, Newricci(tps, delay, qps, param))
			}
		}
	}
	if err == io.EOF {
		err = nil
	}
	if HasErr(err) {
		Ricci = []ricci{}
	}
	return
}

//CheckErr log error
func HasErr(err error) bool {
	if err != nil {
		log.Println(err)
		return true
	}
	return false
}
