SHELL := bash
MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := help

SRC_DIR := golden
TESTS_DIR := tests

.PHONY: install
install: ## Install requirements and golden itself in editable mode
	pip install --upgrade pip
	pip install -e .

.PHONY: develop
develop: ## Install requirements and golden itself in editable mode
	pip install --upgrade pip
	pip install -e .[all]