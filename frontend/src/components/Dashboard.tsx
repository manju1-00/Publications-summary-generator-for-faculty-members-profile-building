import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Button,
} from '@mui/material';
import {
  People,
  LibraryBooks,
  Upload,
  GetApp,
  TrendingUp,
  Assessment,
} from '@mui/icons-material';

const Dashboard: React.FC = () => {
  // Mock data - in real app, this would come from API
  const stats = {
    totalFaculty: 156,
    totalPublications: 2847,
    recentUploads: 23,
    activeExports: 5,
  };

  const recentActivities = [
    { type: 'upload', title: 'Computer Science Department faculty list uploaded', time: '2 hours ago' },
    { type: 'export', title: 'Annual report generated for Engineering department', time: '4 hours ago' },
    { type: 'scan', title: 'Plagiarism scan completed for 45 publications', time: '6 hours ago' },
    { type: 'crawl', title: 'Publication crawl completed for 12 faculty members', time: '1 day ago' },
  ];

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'upload':
        return <Upload color="primary" />;
      case 'export':
        return <GetApp color="secondary" />;
      case 'scan':
        return <Assessment color="error" />;
      case 'crawl':
        return <TrendingUp color="success" />;
      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Welcome to the Academic Publication Management System
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <People color="primary" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4" color="primary">
                    {stats.totalFaculty}
                  </Typography>
                  <Typography color="text.secondary">
                    Total Faculty
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <LibraryBooks color="secondary" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4" color="secondary">
                    {stats.totalPublications}
                  </Typography>
                  <Typography color="text.secondary">
                    Total Publications
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Upload color="success" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4" color="success">
                    {stats.recentUploads}
                  </Typography>
                  <Typography color="text.secondary">
                    Recent Uploads
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <GetApp color="warning" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4" color="warning">
                    {stats.activeExports}
                  </Typography>
                  <Typography color="text.secondary">
                    Active Exports
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box display="flex" flexDirection="column" gap={2}>
              <Button
                variant="contained"
                startIcon={<Upload />}
                fullWidth
              >
                Upload Faculty Data
              </Button>
              <Button
                variant="outlined"
                startIcon={<LibraryBooks />}
                fullWidth
              >
                View Publications
              </Button>
              <Button
                variant="outlined"
                startIcon={<GetApp />}
                fullWidth
              >
                Generate Report
              </Button>
              <Button
                variant="outlined"
                startIcon={<Assessment />}
                fullWidth
              >
                Run Plagiarism Scan
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Activities */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Recent Activities
            </Typography>
            <List>
              {recentActivities.map((activity, index) => (
                <ListItem key={index} divider={index < recentActivities.length - 1}>
                  <ListItemIcon>
                    {getActivityIcon(activity.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={activity.title}
                    secondary={activity.time}
                  />
                  <Chip
                    label={activity.type}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;