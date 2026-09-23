package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

type PredictRequest struct {
	Features map[string]interface{} `json:"features"`
}

type PredictRoastRequest struct {
	ImageBase64 string `json:"image_base64"`
}

type PredictDefectRequest struct {
	ImageBase64 string `json:"image_base64"`
}

type CompareRequest struct {
	CoffeeA PredictRequest `json:"coffeeA"`
	CoffeeB PredictRequest `json:"coffeeB"`
}

var pythonServiceURL string

// Covers the ML service's worst case: two Gemini connect attempts plus one full read.
var pythonClient = &http.Client{Timeout: 50 * time.Second}

func init() {
	godotenv.Load()
	pythonServiceURL = os.Getenv("PYTHON_SERVICE_URL")
	if pythonServiceURL == "" {
		pythonServiceURL = "http://localhost:8001"
	}
}

func setupRouter() *gin.Engine {
	r := gin.Default()

	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	api := r.Group("/api")
	{
		api.POST("/predict", handlePredict)
		api.POST("/explain", handleExplain)
		api.POST("/compare", handleCompare)
		api.POST("/predict-roast", handlePredictRoast)
		api.POST("/predict-defect", handlePredictDefect)
		api.GET("/regions", handleRegions)
		api.GET("/health", handleHealth)
	}
	return r
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	fmt.Printf("Go backend running on port %s\n", port)
	setupRouter().Run(":" + port)
}

func handlePredict(c *gin.Context) {
	proxyPrediction(c, "/predict")
}

func handleExplain(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	proxyPrediction(c, "/explain")
}

func proxyPrediction(c *gin.Context, endpoint string) {
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := forwardToPython(c.Request.Context(), endpoint, req)
	if err != nil {
		writeProxyError(c, err)
		return
	}
	relayJSON(c, resp)
}

func handleCompare(c *gin.Context) {
	var req CompareRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	respA, err := forwardToPython(c.Request.Context(), "/predict", req.CoffeeA)
	if err != nil {
		writeProxyError(c, err)
		return
	}
	if respA.StatusCode != http.StatusOK {
		relayJSON(c, respA)
		return
	}
	defer respA.Body.Close()

	respB, err := forwardToPython(c.Request.Context(), "/predict", req.CoffeeB)
	if err != nil {
		writeProxyError(c, err)
		return
	}
	if respB.StatusCode != http.StatusOK {
		relayJSON(c, respB)
		return
	}
	defer respB.Body.Close()

	var resultA, resultB map[string]interface{}
	if err := json.NewDecoder(respA.Body).Decode(&resultA); err != nil {
		writeProxyError(c, err)
		return
	}
	if err := json.NewDecoder(respB.Body).Decode(&resultB); err != nil {
		writeProxyError(c, err)
		return
	}

	scoreA, okA := resultA["score"].(float64)
	scoreB, okB := resultB["score"].(float64)
	if !okA || !okB {
		writeProxyError(c, errors.New("ML service returned a prediction without a numeric score"))
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"coffeeA": resultA,
		"coffeeB": resultB,
		"diff":    scoreA - scoreB,
	})
}

func handlePredictRoast(c *gin.Context) {
	var req PredictRoastRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := forwardToPython(c.Request.Context(), "/predict-roast", req)
	if err != nil {
		writeProxyError(c, err)
		return
	}
	relayJSON(c, resp)
}

func handlePredictDefect(c *gin.Context) {
	var req PredictDefectRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := forwardToPython(c.Request.Context(), "/predict-defect", req)
	if err != nil {
		writeProxyError(c, err)
		return
	}
	relayJSON(c, resp)
}

func handleRegions(c *gin.Context) {
	resp, err := http.Get(pythonServiceURL + "/regions")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	var regions []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&regions); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, regions)
}

func handleHealth(c *gin.Context) {
	client := http.Client{Timeout: 90 * time.Second}
	pyResp, err := client.Get(pythonServiceURL + "/health")
	pyHealthy := err == nil && pyResp.StatusCode == 200
	if pyResp != nil {
		pyResp.Body.Close()
	}

	c.JSON(http.StatusOK, gin.H{
		"go":           "ok",
		"python":       pyHealthy,
		"modelsLoaded": pyHealthy,
	})
}

func writeProxyError(c *gin.Context, err error) {
	c.Error(err)
	var networkError net.Error
	if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &networkError) && networkError.Timeout()) {
		c.JSON(http.StatusGatewayTimeout, gin.H{"error": "The ML service timed out. Try again."})
		return
	}
	c.JSON(http.StatusBadGateway, gin.H{"error": "The ML service could not complete this request."})
}

func relayJSON(c *gin.Context, resp *http.Response) {
	defer resp.Body.Close()
	const maxResponseBytes = 1 << 20
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		writeProxyError(c, err)
		return
	}
	if len(body) > maxResponseBytes || !json.Valid(body) {
		writeProxyError(c, errors.New("ML service returned an invalid JSON response"))
		return
	}
	if retry, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && retry >= 0 && retry <= 86400 {
		c.Header("Retry-After", strconv.Itoa(retry))
	}
	c.Data(resp.StatusCode, "application/json", body)
}

func forwardToPython(ctx context.Context, endpoint string, payload interface{}) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		pythonServiceURL+endpoint,
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return pythonClient.Do(req)
}
