package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
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
var anthropicAPIKey string

func init() {
	godotenv.Load()
	pythonServiceURL = os.Getenv("PYTHON_SERVICE_URL")
	if pythonServiceURL == "" {
		pythonServiceURL = "http://localhost:8001"
	}
	anthropicAPIKey = os.Getenv("ANTHROPIC_API_KEY")
}

func main() {
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
		api.POST("/compare", handleCompare)
		api.POST("/predict-roast", handlePredictRoast)
		api.POST("/predict-defect", handlePredictDefect)
		api.GET("/regions", handleRegions)
		api.GET("/health", handleHealth)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	fmt.Printf("Go backend running on port %s\n", port)
	r.Run(":" + port)
}

func handlePredict(c *gin.Context) {
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := forwardToPython("/predict", req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	c.JSON(http.StatusOK, result)
}

func handleCompare(c *gin.Context) {
	var req CompareRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	respA, err := forwardToPython("/predict", req.CoffeeA)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer respA.Body.Close()

	respB, err := forwardToPython("/predict", req.CoffeeB)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer respB.Body.Close()

	var resultA, resultB map[string]interface{}
	json.NewDecoder(respA.Body).Decode(&resultA)
	json.NewDecoder(respB.Body).Decode(&resultB)

	scoreA := resultA["score"].(float64)
	scoreB := resultB["score"].(float64)

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

	resp, err := forwardToPython("/predict-roast", req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	c.JSON(http.StatusOK, result)
}

func handlePredictDefect(c *gin.Context) {
	var req PredictDefectRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := forwardToPython("/predict-defect", req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	c.JSON(http.StatusOK, result)
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
		"go":          "ok",
		"python":      pyHealthy,
		"modelsLoaded": pyHealthy,
	})
}

func forwardToPython(endpoint string, payload interface{}) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	return http.Post(
		pythonServiceURL+endpoint,
		"application/json",
		bytes.NewBuffer(body),
	)
}