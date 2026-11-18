# Production Cloud Infrastructure Configuration
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
provider "aws" {
  region = var.aws_region
}
variable "aws_region" {
  type    = string
  default = "us-east-1"
}
# Clean database instance using safe secret references
resource "aws_db_instance" "app_db" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  db_name              = "production_db"
  username             = var.db_username
  password             = var.db_password_secret  # Safe parameterized reference
  skip_final_snapshot  = true
}
